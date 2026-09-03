use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::ModelContextScan;
use codex_rollout::ModelContextScanProgress;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::RolloutItem;
use codex_rollout::ScanOutcome;

use super::LocalThreadStore;
use super::helpers::rollout_path_is_archived;
use super::materialized_resume;
use super::read_thread;
use super::rollout_lineage::RolloutLineage;
use super::rollout_migration::MAX_ROLLOUT_LINE_BYTES;
use super::thread_rollout_resolver;
use crate::LoadModelContextParams;
use crate::ResumeCheckpointOutcome;
use crate::ResumeLoadDiagnostics;
use crate::StoredModelContext;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

#[cfg(test)]
#[path = "model_context_tests.rs"]
mod tests;

/// Loads rollout items needed to reconstruct the latest model-visible context.
///
/// Paginated JSONL rollouts use a reverse scan. When it finds both a usable replacement-
/// history checkpoint and the completed user-turn context needed for resume metadata, the returned
/// replay starts with the canonical `SessionMeta` followed by that newest suffix. When no
/// bounded cutoff is available, the scan continues to the beginning and returns the complete
/// replay it already accumulated.
///
/// Compressed segments are decoded before applying their original JSONL offsets. Legacy rollouts
/// keep the existing full-history path.
///
/// Every read is fenced by the durable source position captured up front. When a private
/// materialized checkpoint matches that fence, resume replays the stored state plus only the
/// appended suffix instead of rescanning the transcript.
pub(super) async fn load_latest_model_context(
    store: &LocalThreadStore,
    params: LoadModelContextParams,
) -> ThreadStoreResult<StoredModelContext> {
    load_model_context(store, params, CheckpointAccess::Resume).await
}

/// Loads source replay for fork and metadata consumers without consuming private resume state.
pub(super) async fn load_latest_model_context_for_replay(
    store: &LocalThreadStore,
    params: LoadModelContextParams,
) -> ThreadStoreResult<StoredModelContext> {
    load_model_context(store, params, CheckpointAccess::SourceReplay).await
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckpointAccess {
    Resume,
    SourceReplay,
}

async fn load_model_context(
    store: &LocalThreadStore,
    params: LoadModelContextParams,
    checkpoint_access: CheckpointAccess,
) -> ThreadStoreResult<StoredModelContext> {
    let requested_path = resolve_model_context_path(store, &params).await?;

    let session_meta = codex_rollout::read_session_meta_line(requested_path.as_path())
        .await
        .map_err(|err| ThreadStoreError::Internal {
            message: format!(
                "failed to read session metadata {}: {err}",
                requested_path.display()
            ),
        })?;
    if session_meta.meta.id != params.thread_id {
        return Err(ThreadStoreError::InvalidRequest {
            message: format!(
                "rollout at {} belongs to thread {}, not {}",
                requested_path.display(),
                session_meta.meta.id,
                params.thread_id
            ),
        });
    }

    let checkpoint_started = Instant::now();
    // Materialization is defined only for store-owned rollouts: the fence, the append-generation
    // journal, and the checkpoint namespace are all keyed by the canonical rollout id. A rollout
    // read from an arbitrary path is still replayable, but it is never materialized.
    let captured_source = if codex_rollout::rollout_id_from_path(requested_path.as_path()).is_some()
    {
        Some(
            materialized_resume::capture_current_source(
                store,
                params.thread_id,
                requested_path.as_path(),
                session_meta.meta.history_mode,
            )
            .await?,
        )
    } else {
        None
    };
    let current_source = captured_source.as_ref().map(|captured| &captured.source);
    let source_fence_bytes = captured_source
        .as_ref()
        .map_or(0, |captured| captured.source_bytes);
    let source_fence_items = captured_source
        .as_ref()
        .map_or(0, |captured| captured.source_items);
    let path = current_source.map_or_else(
        || codex_rollout::plain_rollout_path(requested_path.as_path()),
        |source| source.canonical_rollout_path.clone(),
    );
    let archived = rollout_path_is_archived(store.config.codex_home.as_path(), path.as_path());
    if archived {
        // A probe failure is not evidence of absence. Surface it instead of treating an
        // unreadable artifact directory as a clean archived source.
        let checkpoint_path = materialized_resume::checkpoint_path(store, params.thread_id);
        let checkpoint_exists =
            checkpoint_path
                .try_exists()
                .map_err(|err| ThreadStoreError::Internal {
                    message: format!(
                        "failed to probe materialized resume artifact {}: {err}",
                        checkpoint_path.display()
                    ),
                })?;
        if checkpoint_exists {
            return Err(materialized_resume::invalid_checkpoint(
                "archived source retains a semantically active artifact",
            ));
        }
    }
    if checkpoint_access == CheckpointAccess::Resume
        && !archived
        && let Some(current_source) = current_source
        && let Some(loaded) =
            materialized_resume::load_checkpoint(store, current_source.clone()).await?
    {
        let suffix_started = Instant::now();
        let suffix = read_suffix_segments(
            loaded.suffix_segments.as_slice(),
            loaded.suffix_start_ordinal_exclusive,
            current_source.end_ordinal_exclusive,
        )?;
        let suffix_elapsed = elapsed_millis(suffix_started);
        let suffix_items = u64::try_from(suffix.items.len()).unwrap_or(u64::MAX);
        let mut items = Vec::with_capacity(suffix.items.len().saturating_add(1));
        items.push(RolloutItem::SessionMeta(session_meta));
        items.extend(suffix.items);
        let diagnostics = ResumeLoadDiagnostics {
            outcome: ResumeCheckpointOutcome::Hit,
            source_bytes: loaded
                .source_bytes
                .saturating_add(suffix.bytes)
                .saturating_add(source_fence_bytes),
            source_items: suffix_items.saturating_add(source_fence_items),
            checkpoint_bytes: loaded.checkpoint_bytes,
            checkpoint_items: loaded.checkpoint_items,
            suffix_bytes: suffix.bytes,
            suffix_items,
            scan_elapsed_millis: suffix_elapsed,
            checkpoint_elapsed_millis: elapsed_millis(checkpoint_started),
        };
        record_diagnostics(&diagnostics);
        return Ok(StoredModelContext {
            thread_id: params.thread_id,
            items,
            materialized_resume: Some(loaded.materialized_resume),
            diagnostics,
        });
    }

    let scan_started = Instant::now();
    let mut scanned = match session_meta.meta.history_mode {
        ThreadHistoryMode::Legacy => {
            // Legacy rollouts keep the complete-replay contract. The read is not offset-fenced,
            // so the durable source position is re-verified afterwards: a source that grew during
            // the replay is rejected rather than paired with a fence it no longer matches.
            let items = read_thread::load_history_items(path.as_path()).await?;
            let source_bytes = match current_source {
                Some(current_source) => {
                    materialized_resume::verify_source_unchanged(current_source)?;
                    current_source.end_byte_offset
                }
                None => 0,
            };
            // The complete replay read the whole fenced prefix; report it as such so a bounded
            // resume is distinguishable from a full rescan in diagnostics and tests.
            let source_items = u64::try_from(items.len()).unwrap_or(u64::MAX);
            ScannedModelContext {
                items,
                source_bytes,
                source_items,
            }
        }
        ThreadHistoryMode::Paginated => {
            if params.rollout_path.is_some() {
                ensure_current_paginated_path(store, &params, requested_path.as_path()).await?;
            }
            let lineage = store.resolve_rollout_lineage(params.thread_id).await?;
            if let Some(current_source) = current_source {
                let current_segment = lineage.segments.last().ok_or_else(|| {
                    materialized_resume::invalid_checkpoint("paginated source has no lineage")
                })?;
                if current_segment.rollout_id != current_source.rollout_id {
                    return Err(materialized_resume::invalid_checkpoint(
                        "paginated source fence does not match current lineage",
                    ));
                }
            }
            scan_model_context_from_lineage(
                lineage,
                session_meta.clone(),
                current_source.map(|source| source.end_byte_offset),
            )
            .await?
        }
    };
    if !matches!(scanned.items.first(), Some(RolloutItem::SessionMeta(_))) {
        scanned
            .items
            .insert(0, RolloutItem::SessionMeta(session_meta));
    }

    let diagnostics = ResumeLoadDiagnostics {
        outcome: ResumeCheckpointOutcome::Miss,
        source_bytes: scanned.source_bytes.saturating_add(source_fence_bytes),
        source_items: scanned.source_items.saturating_add(source_fence_items),
        scan_elapsed_millis: elapsed_millis(scan_started),
        checkpoint_elapsed_millis: elapsed_millis(checkpoint_started),
        ..Default::default()
    };
    record_diagnostics(&diagnostics);

    Ok(StoredModelContext {
        thread_id: params.thread_id,
        items: scanned.items,
        materialized_resume: current_source
            .filter(|_| checkpoint_access == CheckpointAccess::Resume && !archived)
            .map(|source| codex_rollout::MaterializedResume {
                source: source.clone(),
                state: None,
                max_state_bytes: None,
            }),
        diagnostics,
    })
}

/// Resolves the exact rollout this model-context load must read.
///
/// A path-addressed resume binds the load to one rollout file; every other caller resolves the
/// thread's current rollout. The two must not be conflated: resolving the logical thread for a
/// path-addressed request would silently fence against a different rollout.
async fn resolve_model_context_path(
    store: &LocalThreadStore,
    params: &LoadModelContextParams,
) -> ThreadStoreResult<PathBuf> {
    if let Some(rollout_path) = params.rollout_path.clone() {
        let path = read_thread::resolve_requested_rollout_path(store, rollout_path).await?;
        if !params.include_archived
            && rollout_path_is_archived(store.config.codex_home.as_path(), path.as_path())
        {
            return Err(ThreadStoreError::InvalidRequest {
                message: format!("thread {} is archived", params.thread_id),
            });
        }
        return Ok(path);
    }

    let resolved = if params.include_archived {
        thread_rollout_resolver::resolve_current_including_archived(store, params.thread_id).await?
    } else {
        thread_rollout_resolver::resolve_current(store, params.thread_id).await?
    };
    resolved
        .map(|resolved| resolved.path)
        .ok_or_else(|| ThreadStoreError::InvalidRequest {
            message: format!("no rollout found for thread id {}", params.thread_id),
        })
}

/// Rejects a path-addressed paginated resume that names anything but the thread's current segment.
///
/// Paginated replay walks the whole lineage, so an older segment's path would produce a replay the
/// requested path does not describe.
async fn ensure_current_paginated_path(
    store: &LocalThreadStore,
    params: &LoadModelContextParams,
    requested_path: &Path,
) -> ThreadStoreResult<()> {
    let current = if params.include_archived {
        thread_rollout_resolver::resolve_current_including_archived(store, params.thread_id).await?
    } else {
        thread_rollout_resolver::resolve_current(store, params.thread_id).await?
    }
    .ok_or_else(|| ThreadStoreError::InvalidRequest {
        message: format!("no rollout found for thread id {}", params.thread_id),
    })?;
    let current_path =
        read_thread::resolve_requested_rollout_path(store, current.path.clone()).await?;
    if codex_rollout::plain_rollout_path(current_path.as_path())
        != codex_rollout::plain_rollout_path(requested_path)
    {
        return Err(ThreadStoreError::InvalidRequest {
            message: format!(
                "paginated rollout at {} is not the current rollout for thread {}",
                requested_path.display(),
                params.thread_id
            ),
        });
    }
    Ok(())
}

/// Loads startup context from a fork's frozen inherited prefix.
pub(super) async fn load_for_fork(
    lineage: RolloutLineage,
    history_base: Option<HistoryPosition>,
) -> ThreadStoreResult<Vec<RolloutItem>> {
    let source_path = lineage
        .segments()
        .last()
        .map(|segment| segment.rollout_path.as_path())
        .ok_or_else(|| ThreadStoreError::Internal {
            message: "fork lineage has no source segment".to_string(),
        })?;
    let session_meta = codex_rollout::read_session_meta_line(source_path)
        .await
        .map_err(|err| ThreadStoreError::Internal {
            message: format!(
                "failed to read session metadata {}: {err}",
                source_path.display()
            ),
        })?;
    match history_base {
        Some(history_base) => {
            let lineage = lineage.truncate_at(history_base).await?;
            scan_model_context_from_lineage(lineage, session_meta, None)
                .await
                .map(|scanned| scanned.items)
        }
        None => Ok(vec![RolloutItem::SessionMeta(session_meta)]),
    }
}

async fn scan_model_context_from_lineage(
    lineage: RolloutLineage,
    session_meta: SessionMetaLine,
    current_end_byte_offset: Option<u64>,
) -> ThreadStoreResult<ScannedModelContext> {
    let scan = tokio::task::spawn_blocking(move || {
        scan_model_context_from_lineage_blocking(&lineage, session_meta, current_end_byte_offset)
    })
    .await
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to join model context scan: {err}"),
    })?;
    match scan {
        Ok(scanned) => Ok(scanned),
        Err(err) => Err(ThreadStoreError::Internal {
            message: format!("failed to scan paginated model context lineage: {err}"),
        }),
    }
}

fn scan_model_context_from_lineage_blocking(
    lineage: &RolloutLineage,
    session_meta: SessionMetaLine,
    current_end_byte_offset: Option<u64>,
) -> io::Result<ScannedModelContext> {
    let mut scan = ModelContextScan::default();
    let mut source_bytes = 0_u64;
    let mut source_items = 0_u64;
    'segments: for (index, segment) in lineage.segments().iter().rev().enumerate() {
        let segment_end_byte_offset = segment.end.map(|end| end.end_byte_offset);
        // Only the newest segment is still open for appends, so only it is fenced by the durable
        // source position captured for this load. Older segments carry their own frozen end.
        let end_byte_offset = if index == 0 {
            segment_end_byte_offset.or(current_end_byte_offset)
        } else {
            segment_end_byte_offset
        };
        let file = codex_rollout::open_rollout_seekable_reader(segment.rollout_path.as_path())?;
        let mut scanner = match end_byte_offset {
            Some(end_byte_offset) => ReverseJsonlScanner::new_at(file, end_byte_offset)?,
            None => ReverseJsonlScanner::new(file)?,
        };
        loop {
            let before = scanner.bytes_read();
            let outcome = scanner.scan_next::<serde_json::Value>()?;
            source_bytes = source_bytes.saturating_add(scanner.bytes_read().saturating_sub(before));
            let Some(outcome) = outcome else {
                break;
            };
            let invalid_record = |err: serde_json::Error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "rollout segment {} contains an invalid record: {err}",
                        segment.rollout_path.display()
                    ),
                )
            };
            let line = match outcome {
                ScanOutcome::Parsed(value) => {
                    codex_rollout::decode_rollout_line(value).map_err(invalid_record)?
                }
                ScanOutcome::Rejected(err) => return Err(invalid_record(err)),
            };
            source_items = source_items.saturating_add(1);
            // Each rollout segment contributes only its local delta. Its session metadata is
            // replaced with the requested thread's canonical SessionMeta after replay.
            if matches!(&line.item, RolloutItem::SessionMeta(_)) {
                break;
            }
            match scan.push(line.item) {
                ModelContextScanProgress::Continue => {}
                ModelContextScanProgress::Complete => break 'segments,
            }
        }
    }

    let canonical_meta = session_meta.clone();
    let mut items = scan.finish(session_meta);
    if !matches!(items.first(), Some(RolloutItem::SessionMeta(_))) {
        items.insert(0, RolloutItem::SessionMeta(canonical_meta));
    }
    Ok(ScannedModelContext {
        items,
        source_bytes,
        source_items,
    })
}

struct ScannedModelContext {
    items: Vec<RolloutItem>,
    source_bytes: u64,
    source_items: u64,
}

struct SuffixRead {
    items: Vec<RolloutItem>,
    bytes: u64,
}

/// Replays exactly the appended records between a checkpoint's fence and the current source end.
///
/// Every deviation - a short read, an ordinal gap, a partial trailing record, or an oversized
/// record - rejects the checkpoint instead of returning a shorter replay.
fn read_suffix_segments(
    segments: &[materialized_resume::SourceSuffixSegment],
    start_ordinal_exclusive: Option<u64>,
    end_ordinal_exclusive: Option<u64>,
) -> ThreadStoreResult<SuffixRead> {
    let mut items = Vec::new();
    let mut consumed = 0_u64;
    let mut next_ordinal = start_ordinal_exclusive;
    for segment in segments {
        let start = segment.start_byte_offset;
        let end = segment.end_byte_offset;
        if start > end {
            return Err(materialized_resume::invalid_checkpoint(format!(
                "checkpoint fence is beyond the source segment end: {}",
                segment.path.display()
            )));
        }
        // The suffix fence addresses original JSONL offsets, so a segment compressed after the
        // checkpoint was published is decoded through the same bounded seekable reader rather
        // than falling back to a full rescan.
        let mut file = codex_rollout::open_rollout_seekable_reader(segment.path.as_path())
            .map_err(|err| ThreadStoreError::Internal {
                message: format!(
                    "failed to open resume suffix {}: {err}",
                    segment.path.display()
                ),
            })?;
        file.seek(SeekFrom::Start(start)).map_err(|err| {
            materialized_resume::invalid_checkpoint(format!("invalid suffix fence: {err}"))
        })?;
        let length = end.saturating_sub(start);
        let mut reader = BufReader::new(file.take(length));
        let mut segment_consumed = 0_u64;
        loop {
            let mut line = Vec::new();
            let read = Read::by_ref(&mut reader)
                .take(MAX_ROLLOUT_LINE_BYTES.saturating_add(1) as u64)
                .read_until(b'\n', &mut line)
                .map_err(|err| {
                    materialized_resume::invalid_checkpoint(format!(
                        "failed to read source suffix: {err}"
                    ))
                })?;
            if read == 0 {
                break;
            }
            if read > MAX_ROLLOUT_LINE_BYTES {
                return Err(materialized_resume::invalid_checkpoint(format!(
                    "source suffix record exceeds the {MAX_ROLLOUT_LINE_BYTES}-byte rollout limit"
                )));
            }
            segment_consumed =
                segment_consumed.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
            if segment_consumed > length {
                return Err(materialized_resume::invalid_checkpoint(
                    "source suffix crossed its stable end fence",
                ));
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            if !line.ends_with(b"\n") {
                return Err(materialized_resume::invalid_checkpoint(
                    "source suffix ends with a partial rollout record",
                ));
            }
            let value = serde_json::from_slice(line.as_slice()).map_err(|err| {
                materialized_resume::invalid_checkpoint(format!("source suffix is corrupt: {err}"))
            })?;
            let rollout_line = codex_rollout::decode_rollout_line(value).map_err(|err| {
                materialized_resume::invalid_checkpoint(format!("source suffix is invalid: {err}"))
            })?;
            if let Some(expected_ordinal) = next_ordinal {
                if rollout_line.ordinal != Some(expected_ordinal) {
                    return Err(materialized_resume::invalid_checkpoint(format!(
                        "source suffix ordinal {:?} does not match expected {expected_ordinal}",
                        rollout_line.ordinal
                    )));
                }
                next_ordinal = Some(expected_ordinal.checked_add(1).ok_or_else(|| {
                    materialized_resume::invalid_checkpoint("source suffix ordinal overflow")
                })?);
            }
            if !matches!(rollout_line.item, RolloutItem::SessionMeta(_)) {
                items.push(rollout_line.item);
            }
        }
        if segment_consumed != length {
            return Err(materialized_resume::invalid_checkpoint(format!(
                "source suffix length mismatch: read {segment_consumed} of {length} bytes"
            )));
        }
        consumed = consumed.saturating_add(segment_consumed);
    }
    if next_ordinal != end_ordinal_exclusive {
        return Err(materialized_resume::invalid_checkpoint(format!(
            "source suffix ended at ordinal {next_ordinal:?}, expected {end_ordinal_exclusive:?}"
        )));
    }
    Ok(SuffixRead {
        items,
        bytes: consumed,
    })
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn record_diagnostics(diagnostics: &ResumeLoadDiagnostics) {
    tracing::info!(
        outcome = ?diagnostics.outcome,
        source_bytes = diagnostics.source_bytes,
        source_items = diagnostics.source_items,
        checkpoint_bytes = diagnostics.checkpoint_bytes,
        checkpoint_items = diagnostics.checkpoint_items,
        suffix_bytes = diagnostics.suffix_bytes,
        suffix_items = diagnostics.suffix_items,
        scan_elapsed_millis = diagnostics.scan_elapsed_millis,
        checkpoint_elapsed_millis = diagnostics.checkpoint_elapsed_millis,
        "loaded materialized resume input"
    );
    let Some(metrics) = codex_otel::global() else {
        return;
    };
    let outcome = match diagnostics.outcome {
        ResumeCheckpointOutcome::Hit => "hit",
        ResumeCheckpointOutcome::Miss => "miss",
    };
    let tags = &[("outcome", outcome)];
    for (name, value) in [
        ("codex.resume.source_bytes", diagnostics.source_bytes),
        ("codex.resume.source_items", diagnostics.source_items),
        (
            "codex.resume.checkpoint_bytes",
            diagnostics.checkpoint_bytes,
        ),
        (
            "codex.resume.checkpoint_items",
            diagnostics.checkpoint_items,
        ),
        ("codex.resume.suffix_bytes", diagnostics.suffix_bytes),
        ("codex.resume.suffix_items", diagnostics.suffix_items),
        (
            "codex.resume.scan_elapsed_millis",
            diagnostics.scan_elapsed_millis,
        ),
        (
            "codex.resume.checkpoint_elapsed_millis",
            diagnostics.checkpoint_elapsed_millis,
        ),
    ] {
        let _ = metrics.histogram(name, i64::try_from(value).unwrap_or(i64::MAX), tags);
    }
    let _ = metrics.counter("codex.resume.checkpoint", /*inc*/ 1, tags);
}
