use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use codex_protocol::ThreadId;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::MATERIALIZED_RESUME_STATE_VERSION;
use codex_rollout::MaterializedResume;
use codex_rollout::MaterializedResumeLineageSegment;
use codex_rollout::MaterializedResumeSource;
use codex_rollout::MaterializedResumeState;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::ScanOutcome;
use sha2::Digest;
use sha2::Sha256;

use super::LocalThreadStore;
use crate::MaterializedResumePublicationFence;
use crate::PublishMaterializedResumeParams;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

#[cfg(test)]
#[path = "materialized_resume_tests.rs"]
mod tests;

const CHECKPOINT_DIRECTORY: &str = "materialized_resume_state_v1";
const FENCE_SAMPLE_BYTES: usize = 64 * 1024;
/// Absolute allocation guard for the private checkpoint artifact. Core supplies a tighter bound
/// derived from the active model context when it publishes the state.
pub(crate) const MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES: u64 = 512 * 1024 * 1024;

pub(super) struct LoadedCheckpoint {
    pub(super) materialized_resume: MaterializedResume,
    pub(super) suffix_segments: Vec<SourceSuffixSegment>,
    pub(super) suffix_start_ordinal_exclusive: Option<u64>,
    pub(super) checkpoint_bytes: u64,
    pub(super) checkpoint_items: u64,
    pub(super) source_bytes: u64,
}

pub(super) struct SourceSuffixSegment {
    pub(super) path: PathBuf,
    pub(super) start_byte_offset: u64,
    pub(super) end_byte_offset: u64,
}

pub(super) struct CapturedSource {
    pub(super) source: MaterializedResumeSource,
    pub(super) source_bytes: u64,
    pub(super) source_items: u64,
}

pub(super) async fn capture_current_source(
    store: &LocalThreadStore,
    thread_id: ThreadId,
    rollout_path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<CapturedSource> {
    let path = if codex_rollout::is_compressed_rollout_path(rollout_path) {
        codex_rollout::materialize_rollout_for_reference(rollout_path)
            .await
            .map_err(|err| ThreadStoreError::Internal {
                message: format!(
                    "failed to materialize resume source {}: {err}",
                    rollout_path.display()
                ),
            })?
    } else {
        rollout_path.to_path_buf()
    };
    let canonical_rollout_path = canonical_existing_path(path.as_path())?;
    let rollout_id = codex_rollout::rollout_id_from_path(canonical_rollout_path.as_path())
        .ok_or_else(|| ThreadStoreError::InvalidRequest {
            message: format!(
                "resume source path has no canonical rollout identity: {}",
                canonical_rollout_path.display()
            ),
        })?;
    let (
        end_byte_offset,
        end_ordinal_exclusive,
        source_modified_unix_nanos,
        terminal_scan_bytes,
        terminal_scan_items,
    ) = stable_position(canonical_rollout_path.as_path(), history_mode)?;
    let (prefix_head_sha256, head_bytes) =
        hash_sample(canonical_rollout_path.as_path(), 0, end_byte_offset)?;
    let tail_start = end_byte_offset.saturating_sub(FENCE_SAMPLE_BYTES as u64);
    let (prefix_tail_sha256, tail_bytes) = hash_sample(
        canonical_rollout_path.as_path(),
        tail_start,
        end_byte_offset,
    )?;
    let mut source_bytes = terminal_scan_bytes
        .saturating_add(head_bytes)
        .saturating_add(tail_bytes);
    let append_generation = super::append_generation::load_current(
        store,
        rollout_id,
        canonical_rollout_path.as_path(),
    )?;
    let lineage = match history_mode {
        ThreadHistoryMode::Legacy => Vec::new(),
        ThreadHistoryMode::Paginated => {
            let resolved = store.resolve_rollout_lineage(thread_id).await?;
            let mut lineage = Vec::with_capacity(resolved.segments().len());
            for segment in resolved.segments() {
                let is_current = segment.rollout_id == rollout_id;
                if codex_rollout::is_compressed_rollout_path(segment.rollout_path.as_path()) {
                    return Err(invalid_checkpoint(
                        "paginated lineage contains a compressed source segment",
                    ));
                }
                let canonical_segment_path =
                    canonical_existing_path(segment.rollout_path.as_path())?;
                let segment_end_byte_offset = segment
                    .end
                    .map_or(end_byte_offset, |end| end.end_byte_offset);
                let (segment_head_sha256, segment_tail_sha256) = if is_current {
                    (prefix_head_sha256.clone(), prefix_tail_sha256.clone())
                } else {
                    let (head, segment_head_bytes) =
                        hash_sample(canonical_segment_path.as_path(), 0, segment_end_byte_offset)?;
                    let tail_start =
                        segment_end_byte_offset.saturating_sub(FENCE_SAMPLE_BYTES as u64);
                    let (tail, segment_tail_bytes) = hash_sample(
                        canonical_segment_path.as_path(),
                        tail_start,
                        segment_end_byte_offset,
                    )?;
                    source_bytes = source_bytes
                        .saturating_add(segment_head_bytes)
                        .saturating_add(segment_tail_bytes);
                    (head, tail)
                };
                lineage.push(MaterializedResumeLineageSegment {
                    rollout_id: segment.rollout_id,
                    canonical_rollout_path: canonical_segment_path.clone(),
                    end_byte_offset: segment_end_byte_offset,
                    end_ordinal_exclusive: segment
                        .end
                        .map(|end| end.end_ordinal_exclusive)
                        .or(is_current.then_some(end_ordinal_exclusive).flatten()),
                    prefix_head_sha256: segment_head_sha256,
                    prefix_tail_sha256: segment_tail_sha256,
                    append_generation: super::append_generation::load_current(
                        store,
                        segment.rollout_id,
                        canonical_segment_path.as_path(),
                    )?,
                });
            }
            lineage
        }
    };
    let verified =
        std::fs::metadata(canonical_rollout_path.as_path()).map_err(source_metadata_error)?;
    if verified.len() != end_byte_offset
        || modified_unix_nanos(verified.modified().map_err(source_metadata_error)?)?
            != source_modified_unix_nanos
    {
        return Err(ThreadStoreError::Conflict {
            message: "resume source changed while its durable fence was captured".to_string(),
        });
    }
    Ok(CapturedSource {
        source: MaterializedResumeSource {
            thread_id,
            rollout_id,
            canonical_rollout_path,
            history_mode,
            end_byte_offset,
            end_ordinal_exclusive,
            modified_unix_nanos: source_modified_unix_nanos,
            prefix_head_sha256,
            prefix_tail_sha256,
            append_generation,
            lineage,
        },
        source_bytes,
        source_items: terminal_scan_items,
    })
}

pub(super) async fn load_checkpoint(
    store: &LocalThreadStore,
    current_source: MaterializedResumeSource,
) -> ThreadStoreResult<Option<LoadedCheckpoint>> {
    let path = checkpoint_path(store, current_source.thread_id);
    let metadata = match std::fs::metadata(path.as_path()) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(invalid_checkpoint(format!("cannot stat artifact: {err}"))),
    };
    let checkpoint_bytes = metadata.len();
    if checkpoint_bytes > MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES {
        return Err(invalid_checkpoint(format!(
            "artifact is {checkpoint_bytes} bytes, exceeding the explicit {MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES}-byte hard limit"
        )));
    }
    let file = File::open(path.as_path())
        .map_err(|err| invalid_checkpoint(format!("cannot open artifact: {err}")))?;
    let mut bytes = Vec::new();
    file.take(MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|err| invalid_checkpoint(format!("cannot read artifact: {err}")))?;
    let checkpoint_bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if checkpoint_bytes > MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES {
        return Err(invalid_checkpoint(format!(
            "artifact exceeds the explicit {MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES}-byte hard limit"
        )));
    }
    let artifact: MaterializedResume = serde_json::from_slice(bytes.as_slice())
        .map_err(|err| invalid_checkpoint(format!("artifact is corrupt: {err}")))?;
    let state = artifact
        .state
        .as_ref()
        .ok_or_else(|| invalid_checkpoint("artifact has no materialized state"))?;
    if state.version != MATERIALIZED_RESUME_STATE_VERSION {
        return Err(invalid_checkpoint(format!(
            "artifact version {} is unsupported; expected {MATERIALIZED_RESUME_STATE_VERSION}",
            state.version
        )));
    }
    let max_state_bytes = artifact
        .max_state_bytes
        .ok_or_else(|| invalid_checkpoint("artifact is missing its model-derived size bound"))?;
    if max_state_bytes > MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES {
        return Err(invalid_checkpoint(format!(
            "artifact declares an invalid {max_state_bytes}-byte state bound"
        )));
    }
    let state_bytes = serialized_state_bytes(state)?;
    if state_bytes > max_state_bytes {
        return Err(invalid_checkpoint(format!(
            "materialized state is {state_bytes} bytes, exceeding its {max_state_bytes}-byte model-context bound"
        )));
    }
    let (source_bytes, suffix_segments) =
        validate_source_prefix(&artifact.source, &current_source)?;
    let suffix_start_ordinal_exclusive = artifact.source.end_ordinal_exclusive;
    Ok(Some(LoadedCheckpoint {
        checkpoint_items: u64::try_from(state.history.len()).unwrap_or(u64::MAX),
        checkpoint_bytes,
        source_bytes,
        suffix_segments,
        suffix_start_ordinal_exclusive,
        materialized_resume: MaterializedResume {
            source: current_source,
            state: artifact.state,
            max_state_bytes: Some(max_state_bytes),
        },
    }))
}

pub(super) async fn publish(
    store: &LocalThreadStore,
    params: PublishMaterializedResumeParams,
) -> ThreadStoreResult<()> {
    if params.state.version != MATERIALIZED_RESUME_STATE_VERSION {
        return Err(invalid_checkpoint(format!(
            "cannot publish materialized state version {}",
            params.state.version
        )));
    }
    if params.max_state_bytes > MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES {
        return Err(invalid_checkpoint(format!(
            "model-derived state bound {} exceeds the explicit {}-byte hard limit",
            params.max_state_bytes, MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES
        )));
    }
    let state_bytes = serialized_state_bytes(&params.state)?;
    if state_bytes > params.max_state_bytes {
        return Err(invalid_checkpoint(format!(
            "materialized state is {state_bytes} bytes, exceeding the {max}-byte model-context bound",
            max = params.max_state_bytes
        )));
    }
    let _writer_guard = store.live_writer_locks.lock(params.thread_id).await;
    let mut current_source = match params.fence {
        MaterializedResumePublicationFence::Loaded(source) => {
            let source = *source;
            if params.thread_id != source.thread_id {
                return Err(invalid_checkpoint("publication thread identity mismatch"));
            }
            let current = capture_current_source(
                store,
                params.thread_id,
                source.canonical_rollout_path.as_path(),
                source.history_mode,
            )
            .await?
            .source;
            validate_exact_source(&source, &current)?;
            current
        }
        MaterializedResumePublicationFence::Current {
            rollout_path,
            history_mode,
        } => {
            capture_current_source(
                store,
                params.thread_id,
                rollout_path.as_path(),
                history_mode,
            )
            .await?
            .source
        }
    };
    ensure_append_generations(store, &mut current_source)?;

    let artifact = MaterializedResume {
        source: current_source,
        state: Some(params.state),
        max_state_bytes: Some(params.max_state_bytes),
    };
    let bytes = serde_json::to_vec(&artifact).map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to encode materialized resume state: {err}"),
    })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MATERIALIZED_RESUME_STATE_HARD_MAX_BYTES {
        return Err(invalid_checkpoint(
            "encoded artifact exceeds the explicit hard limit",
        ));
    }
    atomic_write(
        checkpoint_path(store, params.thread_id).as_path(),
        bytes.as_slice(),
    )
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to publish materialized resume state atomically: {err}"),
    })?;
    bind_append_generation_checkpoints(store, &artifact.source)
}

pub(super) fn remove(store: &LocalThreadStore, thread_id: ThreadId) -> ThreadStoreResult<()> {
    let path = checkpoint_path(store, thread_id);
    match std::fs::remove_file(path.as_path()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(ThreadStoreError::Internal {
            message: format!("failed to remove materialized resume state for {thread_id}: {err}"),
        }),
    }
}

pub(super) fn checkpoint_path(store: &LocalThreadStore, thread_id: ThreadId) -> PathBuf {
    store
        .config
        .codex_home
        .join(CHECKPOINT_DIRECTORY)
        .join(format!("{thread_id}.json"))
}

fn stable_position(
    path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<(u64, Option<u64>, u64, u64, u64)> {
    let before = std::fs::metadata(path).map_err(source_metadata_error)?;
    let end_byte_offset = before.len();
    let (end_ordinal_exclusive, terminal_scan_bytes, terminal_scan_items) = match history_mode {
        ThreadHistoryMode::Legacy => (None, 0, 0),
        ThreadHistoryMode::Paginated => {
            let file = File::open(path).map_err(source_metadata_error)?;
            let mut scanner = ReverseJsonlScanner::new(file)
                .map_err(source_metadata_error)?
                .with_strict_max_record_bytes(codex_rollout::MAX_ROLLOUT_LINE_BYTES);
            let line = match scanner.scan_next_rollout_line().map_err(|err| {
                invalid_checkpoint(format!("source terminal record cannot be fenced: {err}"))
            })? {
                Some(ScanOutcome::Parsed(line)) => line,
                Some(ScanOutcome::Rejected(err)) => {
                    return Err(invalid_checkpoint(format!(
                        "source terminal record is corrupt: {err}"
                    )));
                }
                None => return Err(invalid_checkpoint("source rollout is empty")),
            };
            let ordinal = line.ordinal.ok_or_else(|| {
                invalid_checkpoint("paginated source terminal record has no ordinal")
            })?;
            let end_ordinal_exclusive = ordinal
                .checked_add(1)
                .ok_or_else(|| invalid_checkpoint("paginated source ordinal overflow"))?;
            (Some(end_ordinal_exclusive), scanner.bytes_read(), 1)
        }
    };
    let after = std::fs::metadata(path).map_err(source_metadata_error)?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(ThreadStoreError::Conflict {
            message: "resume source changed while its durable position was captured".to_string(),
        });
    }
    Ok((
        end_byte_offset,
        end_ordinal_exclusive,
        modified_unix_nanos(after.modified().map_err(source_metadata_error)?)?,
        terminal_scan_bytes,
        terminal_scan_items,
    ))
}

fn validate_source_prefix(
    stored: &MaterializedResumeSource,
    current: &MaterializedResumeSource,
) -> ThreadStoreResult<(u64, Vec<SourceSuffixSegment>)> {
    if stored.thread_id != current.thread_id || stored.history_mode != current.history_mode {
        return Err(invalid_checkpoint(
            "source thread identity or history mode mismatch",
        ));
    }
    match stored.history_mode {
        ThreadHistoryMode::Legacy => {
            if stored.rollout_id != current.rollout_id
                || stored.canonical_rollout_path != current.canonical_rollout_path
            {
                return Err(invalid_checkpoint(
                    "source path or canonical rollout identity mismatch",
                ));
            }
            validate_generation(
                stored.append_generation.as_ref(),
                current.append_generation.as_ref(),
            )?;
            validate_suffix_bounds(
                stored.end_byte_offset,
                stored.end_ordinal_exclusive,
                current.end_byte_offset,
                current.end_ordinal_exclusive,
            )?;
            Ok((
                0,
                vec![SourceSuffixSegment {
                    path: current.canonical_rollout_path.clone(),
                    start_byte_offset: stored.end_byte_offset,
                    end_byte_offset: current.end_byte_offset,
                }],
            ))
        }
        ThreadHistoryMode::Paginated => validate_paginated_source_prefix(stored, current),
    }
}

fn validate_exact_source(
    stored: &MaterializedResumeSource,
    current: &MaterializedResumeSource,
) -> ThreadStoreResult<()> {
    validate_source_identity(stored, current)?;
    if stored.end_byte_offset != current.end_byte_offset
        || stored.end_ordinal_exclusive != current.end_ordinal_exclusive
        || stored.modified_unix_nanos != current.modified_unix_nanos
        || stored.prefix_head_sha256 != current.prefix_head_sha256
        || stored.prefix_tail_sha256 != current.prefix_tail_sha256
        || stored.append_generation != current.append_generation
        || stored.lineage != current.lineage
    {
        return Err(ThreadStoreError::Conflict {
            message: "resume source changed before materialized state publication".to_string(),
        });
    }
    Ok(())
}

fn validate_source_identity(
    stored: &MaterializedResumeSource,
    current: &MaterializedResumeSource,
) -> ThreadStoreResult<()> {
    if stored.thread_id != current.thread_id
        || stored.rollout_id != current.rollout_id
        || stored.canonical_rollout_path != current.canonical_rollout_path
        || stored.history_mode != current.history_mode
    {
        return Err(invalid_checkpoint(
            "source path or canonical rollout identity mismatch",
        ));
    }
    Ok(())
}

fn validate_paginated_source_prefix(
    stored: &MaterializedResumeSource,
    current: &MaterializedResumeSource,
) -> ThreadStoreResult<(u64, Vec<SourceSuffixSegment>)> {
    if stored.lineage.is_empty() || current.lineage.len() < stored.lineage.len() {
        return Err(invalid_checkpoint(
            "source lineage moved behind the checkpoint",
        ));
    }
    let stored_last_index = stored.lineage.len().saturating_sub(1);
    for (index, (stored_segment, current_segment)) in
        stored.lineage.iter().zip(&current.lineage).enumerate()
    {
        if stored_segment.rollout_id != current_segment.rollout_id
            || stored_segment.canonical_rollout_path != current_segment.canonical_rollout_path
            || (index != stored_last_index
                && (stored_segment.end_byte_offset != current_segment.end_byte_offset
                    || stored_segment.end_ordinal_exclusive
                        != current_segment.end_ordinal_exclusive
                    || stored_segment.prefix_head_sha256 != current_segment.prefix_head_sha256
                    || stored_segment.prefix_tail_sha256 != current_segment.prefix_tail_sha256))
        {
            return Err(invalid_checkpoint("source lineage identity mismatch"));
        }
        validate_generation(
            stored_segment.append_generation.as_ref(),
            current_segment.append_generation.as_ref(),
        )?;
    }
    let stored_last = &stored.lineage[stored_last_index];
    if stored.rollout_id != stored_last.rollout_id
        || stored.canonical_rollout_path != stored_last.canonical_rollout_path
        || stored.end_byte_offset != stored_last.end_byte_offset
        || stored.end_ordinal_exclusive != stored_last.end_ordinal_exclusive
        || stored.append_generation != stored_last.append_generation
    {
        return Err(invalid_checkpoint(
            "checkpoint source does not match its terminal lineage segment",
        ));
    }
    let current_last = current
        .lineage
        .last()
        .ok_or_else(|| invalid_checkpoint("current paginated source has no terminal lineage"))?;
    if current.rollout_id != current_last.rollout_id
        || current.canonical_rollout_path != current_last.canonical_rollout_path
        || current.end_byte_offset != current_last.end_byte_offset
        || current.end_ordinal_exclusive != current_last.end_ordinal_exclusive
        || current.append_generation != current_last.append_generation
    {
        return Err(invalid_checkpoint(
            "current source does not match its terminal lineage segment",
        ));
    }
    let current_at_checkpoint = &current.lineage[stored_last_index];
    validate_suffix_bounds(
        stored.end_byte_offset,
        stored.end_ordinal_exclusive,
        current_at_checkpoint.end_byte_offset,
        current_at_checkpoint.end_ordinal_exclusive,
    )?;
    let mut suffix_segments = vec![SourceSuffixSegment {
        path: current_at_checkpoint.canonical_rollout_path.clone(),
        start_byte_offset: stored.end_byte_offset,
        end_byte_offset: current_at_checkpoint.end_byte_offset,
    }];
    suffix_segments.extend(
        current
            .lineage
            .iter()
            .skip(stored.lineage.len())
            .map(|segment| SourceSuffixSegment {
                path: segment.canonical_rollout_path.clone(),
                start_byte_offset: 0,
                end_byte_offset: segment.end_byte_offset,
            }),
    );
    Ok((0, suffix_segments))
}

fn validate_generation(
    stored: Option<&codex_rollout::MaterializedResumeAppendGeneration>,
    current: Option<&codex_rollout::MaterializedResumeAppendGeneration>,
) -> ThreadStoreResult<()> {
    let stored = stored.ok_or_else(|| invalid_checkpoint("checkpoint has no append generation"))?;
    let current = current.ok_or_else(|| invalid_checkpoint("source has no append generation"))?;
    super::append_generation::validate_descendant(stored, current)
}

fn validate_suffix_bounds(
    stored_byte_offset: u64,
    stored_ordinal_exclusive: Option<u64>,
    current_byte_offset: u64,
    current_ordinal_exclusive: Option<u64>,
) -> ThreadStoreResult<()> {
    if current_byte_offset < stored_byte_offset {
        return Err(invalid_checkpoint("source was truncated before the fence"));
    }
    if current_ordinal_exclusive
        .zip(stored_ordinal_exclusive)
        .is_some_and(|(current, stored)| current < stored)
    {
        return Err(invalid_checkpoint("source ordinal moved behind the fence"));
    }
    Ok(())
}

fn ensure_append_generations(
    store: &LocalThreadStore,
    source: &mut MaterializedResumeSource,
) -> ThreadStoreResult<()> {
    source.append_generation = Some(super::append_generation::bootstrap_current(
        store,
        source.rollout_id,
        source.canonical_rollout_path.as_path(),
        source.history_mode,
    )?);
    for segment in &mut source.lineage {
        segment.append_generation = Some(super::append_generation::bootstrap_current(
            store,
            segment.rollout_id,
            segment.canonical_rollout_path.as_path(),
            ThreadHistoryMode::Paginated,
        )?);
    }
    Ok(())
}

fn bind_append_generation_checkpoints(
    store: &LocalThreadStore,
    source: &MaterializedResumeSource,
) -> ThreadStoreResult<()> {
    super::append_generation::bind_checkpoint(
        store,
        source.rollout_id,
        source.canonical_rollout_path.as_path(),
    )?;
    for segment in &source.lineage {
        if segment.rollout_id != source.rollout_id {
            super::append_generation::bind_checkpoint(
                store,
                segment.rollout_id,
                segment.canonical_rollout_path.as_path(),
            )?;
        }
    }
    Ok(())
}

fn hash_sample(path: &Path, start: u64, end: u64) -> ThreadStoreResult<(String, u64)> {
    let bounded_start = start.min(end);
    let sample_end = bounded_start
        .saturating_add(FENCE_SAMPLE_BYTES as u64)
        .min(end);
    let length = sample_end.saturating_sub(bounded_start);
    let mut file = File::open(path).map_err(source_metadata_error)?;
    file.seek(SeekFrom::Start(bounded_start))
        .map_err(source_metadata_error)?;
    let mut bytes = vec![
        0;
        usize::try_from(length).map_err(|_| {
            invalid_checkpoint("source fence sample exceeds addressable memory")
        })?
    ];
    file.read_exact(bytes.as_mut_slice())
        .map_err(source_metadata_error)?;
    Ok((format!("{:x}", Sha256::digest(bytes)), length))
}

fn canonical_existing_path(path: &Path) -> ThreadStoreResult<PathBuf> {
    std::fs::canonicalize(path).map_err(|err| ThreadStoreError::Internal {
        message: format!(
            "failed to canonicalize resume source {}: {err}",
            path.display()
        ),
    })
}

fn serialized_state_bytes(state: &MaterializedResumeState) -> ThreadStoreResult<u64> {
    let bytes = serde_json::to_vec(state).map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to measure materialized resume state: {err}"),
    })?;
    Ok(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
}

fn modified_unix_nanos(modified: SystemTime) -> ThreadStoreResult<u64> {
    let nanos = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid_checkpoint("source modified time predates the Unix epoch"))?
        .as_nanos();
    u64::try_from(nanos).map_err(|_| invalid_checkpoint("source modified time overflow"))
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("checkpoint path has no parent: {}", path.display()),
        )
    })?;
    std::fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    temporary.write_all(bytes)?;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|err| err.error)?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn source_metadata_error(err: io::Error) -> ThreadStoreError {
    ThreadStoreError::Internal {
        message: format!("failed to inspect materialized resume source: {err}"),
    }
}

pub(crate) fn invalid_checkpoint(reason: impl std::fmt::Display) -> ThreadStoreError {
    let reason = reason.to_string();
    tracing::error!(reason, "materialized resume state is invalid");
    if let Some(metrics) = codex_otel::global() {
        let _ = metrics.counter(
            "codex.resume.checkpoint",
            /*inc*/ 1,
            &[("outcome", "invalid")],
        );
    }
    ThreadStoreError::InvalidRequest {
        message: format!("codex_resume_state_needs_compaction: {reason}"),
    }
}
