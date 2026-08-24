use std::fs::File;
use std::io;
use std::path::PathBuf;

use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::ModelContextScan;
use codex_rollout::ModelContextScanProgress;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutLine;
use codex_rollout::ScanOutcome;

use super::LocalThreadStore;
use super::helpers::rollout_path_is_archived;
use super::read_thread;
use super::rollout_lineage::RolloutLineage;
use super::thread_rollout_resolver;
use crate::LoadModelContextParams;
use crate::StoredModelContext;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

#[cfg(test)]
#[path = "model_context_tests.rs"]
mod tests;

/// Loads rollout items needed to reconstruct the latest model-visible context.
///
/// Plain JSONL rollouts use a reverse scan. When it finds both a usable replacement-
/// history checkpoint and the completed user-turn context needed for resume metadata, the returned
/// replay starts with the canonical `SessionMeta` followed by that newest suffix. When no
/// bounded cutoff is available, the scan continues to the beginning and returns the complete
/// replay it already accumulated.
///
/// Compressed rollout shapes keep the existing full-history path because they are not seekable.
pub(super) async fn load_latest_model_context(
    store: &LocalThreadStore,
    params: LoadModelContextParams,
) -> ThreadStoreResult<StoredModelContext> {
    let path = resolve_model_context_path(store, &params).await?;

    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .map_err(|err| ThreadStoreError::Internal {
            message: format!("failed to read session metadata {}: {err}", path.display()),
        })?;
    if session_meta.meta.id != params.thread_id {
        return Err(ThreadStoreError::InvalidRequest {
            message: format!(
                "rollout at {} belongs to thread {}, not {}",
                path.display(),
                session_meta.meta.id,
                params.thread_id
            ),
        });
    }

    let compressed = path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .is_some_and(|file_name| file_name.ends_with(".jsonl.zst"));
    let items = if compressed {
        read_thread::load_history_items(path.as_path()).await?
    } else {
        match session_meta.meta.history_mode {
            ThreadHistoryMode::Legacy => {
                scan_model_context_from_rollout(path, session_meta).await?
            }
            ThreadHistoryMode::Paginated => {
                if params.rollout_path.is_some() {
                    ensure_current_paginated_path(store, &params, path.as_path()).await?;
                }
                let lineage = store.resolve_rollout_lineage(params.thread_id).await?;
                scan_model_context_from_lineage(lineage, session_meta).await?
            }
        }
    };

    Ok(StoredModelContext {
        thread_id: params.thread_id,
        items,
    })
}

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

async fn ensure_current_paginated_path(
    store: &LocalThreadStore,
    params: &LoadModelContextParams,
    requested_path: &std::path::Path,
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
    if current_path != requested_path {
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
            scan_model_context_from_lineage(lineage, session_meta).await
        }
        None => Ok(vec![RolloutItem::SessionMeta(session_meta)]),
    }
}

async fn scan_model_context_from_lineage(
    lineage: RolloutLineage,
    session_meta: SessionMetaLine,
) -> ThreadStoreResult<Vec<RolloutItem>> {
    let scan = tokio::task::spawn_blocking(move || {
        scan_model_context_from_lineage_blocking(&lineage, session_meta)
    })
    .await
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to join model context scan: {err}"),
    })?;
    match scan {
        Ok(items) => Ok(items),
        Err(err) => Err(ThreadStoreError::Internal {
            message: format!("failed to scan model context: {err}"),
        }),
    }
}

async fn scan_model_context_from_rollout(
    path: PathBuf,
    session_meta: SessionMetaLine,
) -> ThreadStoreResult<Vec<RolloutItem>> {
    let scan = tokio::task::spawn_blocking(move || {
        scan_model_context_from_segments_blocking(
            vec![(path, None, /*stop_at_session_meta*/ false)],
            session_meta,
        )
    })
    .await
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to join model context scan: {err}"),
    })?;
    scan.map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to scan model context: {err}"),
    })
}

fn scan_model_context_from_lineage_blocking(
    lineage: &RolloutLineage,
    session_meta: SessionMetaLine,
) -> io::Result<Vec<RolloutItem>> {
    let segments = lineage
        .segments()
        .iter()
        .rev()
        .map(|segment| {
            (
                segment.rollout_path.clone(),
                segment.end.map(|end| end.end_byte_offset),
                /*stop_at_session_meta*/ true,
            )
        })
        .collect();
    scan_model_context_from_segments_blocking(segments, session_meta)
}

fn scan_model_context_from_segments_blocking(
    segments: Vec<(PathBuf, Option<u64>, bool)>,
    session_meta: SessionMetaLine,
) -> io::Result<Vec<RolloutItem>> {
    let mut scan = ModelContextScan::default();
    'segments: for (rollout_path, end_byte_offset, stop_at_session_meta) in segments {
        let file = File::open(rollout_path)?;
        let mut scanner = match end_byte_offset {
            Some(end_byte_offset) => ReverseJsonlScanner::new_at(file, end_byte_offset)?,
            None => ReverseJsonlScanner::new(file)?,
        };
        while let Some(outcome) = scanner.scan_next::<RolloutLine>()? {
            let ScanOutcome::Parsed(line) = outcome else {
                continue;
            };
            // Each rollout segment contributes only its local delta. Its session metadata is
            // replaced with the requested thread's canonical SessionMeta after replay.
            if stop_at_session_meta && matches!(&line.item, RolloutItem::SessionMeta(_)) {
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
    Ok(items)
}
