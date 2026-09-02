use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use codex_protocol::ThreadId;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::MaterializedResumeAppendGeneration;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::RolloutItem;
use codex_rollout::ScanOutcome;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

use super::LocalThreadStore;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

#[cfg(windows)]
compile_error!(
    "codex_resume_state_needs_platform_integrity: this release cannot create sessions on Windows until the rollout fence has native file identity and change-time support"
);

const APPEND_GENERATION_VERSION: u32 = 5;
const APPEND_GENERATION_DIRECTORY: &str = "rollout_append_generation_v5";
const MAX_APPEND_GENERATION_BYTES: u64 = 1024 * 1024;
const MAX_CHECKPOINT_ANCHORS: usize = 4_096;
const FENCE_SAMPLE_BYTES: usize = 64 * 1024;

pub(super) type PlatformFileGeneration = (String, String);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AppendGenerationIo {
    pub(super) source_bytes: u64,
    pub(super) suffix_bytes: u64,
    pub(super) suffix_items: u64,
}

impl AppendGenerationIo {
    fn add_source_bytes(&mut self, source_bytes: u64) {
        self.source_bytes = self.source_bytes.saturating_add(source_bytes);
    }

    fn add_suffix(&mut self, suffix_bytes: u64, suffix_items: u64) {
        self.source_bytes = self.source_bytes.saturating_add(suffix_bytes);
        self.suffix_bytes = self.suffix_bytes.saturating_add(suffix_bytes);
        self.suffix_items = self.suffix_items.saturating_add(suffix_items);
    }

    fn merge(&mut self, other: Self) {
        self.source_bytes = self.source_bytes.saturating_add(other.source_bytes);
        self.suffix_bytes = self.suffix_bytes.saturating_add(other.suffix_bytes);
        self.suffix_items = self.suffix_items.saturating_add(other.suffix_items);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AppendGenerationStart {
    pub(super) started: bool,
    pub(super) io: AppendGenerationIo,
}

pub(super) struct LoadedAppendGeneration {
    pub(super) generation: Option<MaterializedResumeAppendGeneration>,
    pub(super) io: AppendGenerationIo,
}

#[derive(Clone, Copy)]
enum PendingRecoveryContext {
    Continue,
    FinishCurrentAppend,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AppendGenerationJournal {
    version: u32,
    rollout_id: ThreadId,
    canonical_rollout_path: PathBuf,
    generation_id: String,
    stable: StableGeneration,
    checkpoint_anchors: Vec<CheckpointAnchor>,
    pending: Option<PendingAppend>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StableGeneration {
    generation: u64,
    chain_sha256: String,
    position: SourcePosition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CheckpointAnchor {
    anchor_id: String,
    checkpoint_thread_id: ThreadId,
    generation: u64,
    chain_sha256: String,
    descendant_generation: u64,
    descendant_chain_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PendingAppend {
    generation: u64,
    evidence: PendingAppendEvidence,
    history_mode: ThreadHistoryMode,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PendingAppendEvidence {
    ExactItems {
        item_count: u64,
        items_sha256: String,
    },
    Sync,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SourcePosition {
    end_byte_offset: u64,
    end_ordinal_exclusive: Option<u64>,
    modified_unix_nanos: u64,
    file_identity: String,
    change_marker: String,
    prefix_head_sha256: String,
    prefix_middle_sha256: String,
    prefix_tail_sha256: String,
}

pub(super) fn begin_append(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    history_mode: ThreadHistoryMode,
    items: &[RolloutItem],
) -> ThreadStoreResult<AppendGenerationStart> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        if super::materialized_resume::checkpoint_path(store, checkpoint_thread_id)
            .try_exists()
            .map_err(source_error)?
        {
            return Err(invalid(format!(
                "append generation is missing for rollout {rollout_id} while a materialized resume artifact exists"
            )));
        }
        return Ok(AppendGenerationStart::default());
    };
    let mut io = recover_pending(store, &mut journal, PendingRecoveryContext::Continue)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    io.merge(validate_current_position(&journal)?);
    let started = begin_pending_append(
        store,
        &mut journal,
        history_mode,
        PendingAppendEvidence::ExactItems {
            item_count: u64::try_from(items.len()).unwrap_or(u64::MAX),
            items_sha256: hash_items(items)?,
        },
    )?;
    Ok(AppendGenerationStart { started, io })
}

pub(super) fn begin_sync(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<AppendGenerationStart> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        if super::materialized_resume::checkpoint_path(store, checkpoint_thread_id)
            .try_exists()
            .map_err(source_error)?
        {
            return Err(invalid(format!(
                "append generation is missing for rollout {rollout_id} while a materialized resume artifact exists"
            )));
        }
        return Ok(AppendGenerationStart::default());
    };
    let mut io = recover_pending(store, &mut journal, PendingRecoveryContext::Continue)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    io.merge(validate_current_position(&journal)?);
    let started = begin_pending_append(
        store,
        &mut journal,
        history_mode,
        PendingAppendEvidence::Sync,
    )?;
    Ok(AppendGenerationStart { started, io })
}

fn begin_pending_append(
    store: &LocalThreadStore,
    journal: &mut AppendGenerationJournal,
    history_mode: ThreadHistoryMode,
    evidence: PendingAppendEvidence,
) -> ThreadStoreResult<bool> {
    let generation = journal
        .stable
        .generation
        .checked_add(1)
        .ok_or_else(|| invalid("append generation overflow"))?;
    journal.pending = Some(PendingAppend {
        generation,
        evidence,
        history_mode,
    });
    write_journal(store, journal)?;
    Ok(true)
}

pub(super) fn finish_append(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
) -> ThreadStoreResult<AppendGenerationIo> {
    let mut journal = load_journal(store, rollout_id)?
        .ok_or_else(|| invalid("append generation disappeared during canonical append"))?;
    recover_pending(
        store,
        &mut journal,
        PendingRecoveryContext::FinishCurrentAppend,
    )
}

pub(super) fn load_current(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
    rollout_path: &Path,
) -> ThreadStoreResult<Option<MaterializedResumeAppendGeneration>> {
    Ok(load_current_with_io(store, rollout_id, rollout_path)?.generation)
}

pub(super) fn load_current_with_io(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
    rollout_path: &Path,
) -> ThreadStoreResult<LoadedAppendGeneration> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        return Ok(LoadedAppendGeneration {
            generation: None,
            io: AppendGenerationIo::default(),
        });
    };
    let mut io = recover_pending(store, &mut journal, PendingRecoveryContext::Continue)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    io.merge(validate_current_position(&journal)?);
    Ok(LoadedAppendGeneration {
        generation: Some(materialized_generation(&journal, None)),
        io,
    })
}

pub(super) fn bootstrap_current(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
    rollout_path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<MaterializedResumeAppendGeneration> {
    if let Some(generation) = load_current(store, rollout_id, rollout_path)? {
        return Ok(generation);
    }
    let canonical_rollout_path = canonical_existing_path(rollout_path)?;
    let position = source_position(canonical_rollout_path.as_path(), history_mode)?.position;
    let generation_id = ThreadId::new().to_string();
    let chain_sha256 = genesis_chain(&generation_id, rollout_id, &position)?;
    let journal = AppendGenerationJournal {
        version: APPEND_GENERATION_VERSION,
        rollout_id,
        canonical_rollout_path,
        generation_id,
        stable: StableGeneration {
            generation: 0,
            chain_sha256,
            position,
        },
        checkpoint_anchors: Vec::new(),
        pending: None,
    };
    write_journal(store, &journal)?;
    Ok(materialized_generation(&journal, None))
}

pub(super) fn validate_checkpoint_descendant(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    stored: &MaterializedResumeAppendGeneration,
) -> ThreadStoreResult<()> {
    let mut journal = load_journal(store, rollout_id)?
        .ok_or_else(|| invalid("source has no append generation"))?;
    recover_pending(store, &mut journal, PendingRecoveryContext::Continue)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    if stored.generation_id != journal.generation_id
        || journal.stable.generation < stored.generation
    {
        return Err(invalid(
            "rollout append generation is not a descendant of the checkpoint",
        ));
    }
    let anchor_id = stored
        .checkpoint_anchor_id
        .as_deref()
        .ok_or_else(|| invalid("checkpoint append generation has no ancestry anchor"))?;
    let anchor = journal
        .checkpoint_anchors
        .iter()
        .find(|anchor| anchor.anchor_id == anchor_id)
        .ok_or_else(|| {
            invalid("rollout append generation is missing the checkpoint ancestry anchor")
        })?;
    if anchor.checkpoint_thread_id != checkpoint_thread_id
        || anchor.generation != stored.generation
        || anchor.chain_sha256 != stored.chain_sha256
        || anchor.descendant_generation != journal.stable.generation
        || anchor.descendant_chain_sha256 != journal.stable.chain_sha256
    {
        return Err(invalid(
            "rollout append generation ancestry anchor does not reproduce the current chain",
        ));
    }
    Ok(())
}

pub(super) fn prepare_checkpoint_anchor(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    retained_anchor_id: Option<&str>,
) -> ThreadStoreResult<MaterializedResumeAppendGeneration> {
    let mut journal = load_journal(store, rollout_id)?
        .ok_or_else(|| invalid("append generation disappeared before checkpoint publication"))?;
    recover_pending(store, &mut journal, PendingRecoveryContext::Continue)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    journal.checkpoint_anchors.retain(|anchor| {
        anchor.checkpoint_thread_id != checkpoint_thread_id
            || retained_anchor_id == Some(anchor.anchor_id.as_str())
    });
    if journal.checkpoint_anchors.len() >= MAX_CHECKPOINT_ANCHORS {
        return Err(invalid(format!(
            "rollout append generation reached its explicit {MAX_CHECKPOINT_ANCHORS}-checkpoint anchor limit"
        )));
    }
    let anchor_id = ThreadId::new().to_string();
    journal.checkpoint_anchors.push(CheckpointAnchor {
        anchor_id: anchor_id.clone(),
        checkpoint_thread_id,
        generation: journal.stable.generation,
        chain_sha256: journal.stable.chain_sha256.clone(),
        descendant_generation: journal.stable.generation,
        descendant_chain_sha256: journal.stable.chain_sha256.clone(),
    });
    write_journal(store, &journal)?;
    Ok(materialized_generation(&journal, Some(anchor_id)))
}

pub(super) fn discard_stale_checkpoint_anchors(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    retained_anchor_id: Option<&str>,
) -> ThreadStoreResult<()> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        return Ok(());
    };
    if journal.rollout_id != rollout_id || journal.canonical_rollout_path != rollout_path {
        return Err(invalid(
            "rollout append generation identity mismatch during checkpoint cleanup",
        ));
    }
    let original_len = journal.checkpoint_anchors.len();
    journal.checkpoint_anchors.retain(|anchor| {
        anchor.checkpoint_thread_id != checkpoint_thread_id
            || retained_anchor_id == Some(anchor.anchor_id.as_str())
    });
    if journal.checkpoint_anchors.len() != original_len {
        write_journal(store, &journal)?;
    }
    Ok(())
}

pub(super) fn remove(store: &LocalThreadStore, rollout_id: ThreadId) -> ThreadStoreResult<()> {
    let path = journal_path(store, rollout_id);
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(ThreadStoreError::Internal {
            message: format!("failed to remove rollout append generation for {rollout_id}: {err}"),
        }),
    }
}

fn recover_pending(
    store: &LocalThreadStore,
    journal: &mut AppendGenerationJournal,
    context: PendingRecoveryContext,
) -> ThreadStoreResult<AppendGenerationIo> {
    let Some(pending) = journal.pending.clone() else {
        return Ok(AppendGenerationIo::default());
    };
    let current_end = std::fs::metadata(journal.canonical_rollout_path.as_path())
        .map_err(source_error)?
        .len();
    let mut io = AppendGenerationIo::default();
    if current_end < journal.stable.position.end_byte_offset {
        return Err(invalid("pending append truncated the stored source prefix"));
    }
    if current_end == journal.stable.position.end_byte_offset {
        let current = source_position(
            journal.canonical_rollout_path.as_path(),
            pending.history_mode,
        )?;
        io.add_source_bytes(current.source_bytes);
        if !positions_equal(&current.position, &journal.stable.position) {
            return Err(invalid(
                "source changed without completing its pending append",
            ));
        }
        journal.pending = None;
        write_journal(store, journal)?;
        return Ok(io);
    }
    io.add_source_bytes(validate_stored_prefix(
        journal.canonical_rollout_path.as_path(),
        &journal.stable.position,
        pending.history_mode,
    )?);
    let suffix = match summarize_suffix(
        journal.canonical_rollout_path.as_path(),
        journal.stable.position.end_byte_offset,
        current_end,
        journal.stable.position.end_ordinal_exclusive,
        pending.history_mode,
    ) {
        Ok(suffix) => suffix,
        Err(err) => {
            rollback_pending_suffix(store, journal, pending.history_mode, &mut io)?;
            return pending_rollback_result(context, io, err.to_string());
        }
    };
    io.add_suffix(suffix.bytes_read, suffix.item_count);
    let evidence_error = match &pending.evidence {
        PendingAppendEvidence::ExactItems { item_count, .. }
            if suffix.item_count != *item_count =>
        {
            Some(format!(
                "pending append wrote {} items; expected {item_count}",
                suffix.item_count
            ))
        }
        PendingAppendEvidence::ExactItems { items_sha256, .. }
            if suffix.items_sha256 != *items_sha256 =>
        {
            Some("pending append items do not match the canonical write intent".to_string())
        }
        PendingAppendEvidence::Sync
            if suffix.item_count != 0 && matches!(context, PendingRecoveryContext::Continue) =>
        {
            Some("interrupted sync left an unverified durable suffix".to_string())
        }
        PendingAppendEvidence::ExactItems { .. } | PendingAppendEvidence::Sync => None,
    };
    if let Some(reason) = evidence_error {
        rollback_pending_suffix(store, journal, pending.history_mode, &mut io)?;
        return pending_rollback_result(context, io, reason);
    }
    let verified_current = source_position(
        journal.canonical_rollout_path.as_path(),
        pending.history_mode,
    )?;
    io.add_source_bytes(verified_current.source_bytes);
    if verified_current.position.end_byte_offset != current_end
        || verified_current.position.end_ordinal_exclusive != suffix.end_ordinal_exclusive
    {
        return Err(ThreadStoreError::Conflict {
            message: "rollout changed while its pending suffix was inspected".to_string(),
        });
    }
    let chain_sha256 = advance_chain(journal.stable.chain_sha256.as_str(), suffix.sha256.as_str());
    for anchor in &mut journal.checkpoint_anchors {
        anchor.descendant_generation = pending.generation;
        anchor.descendant_chain_sha256 = advance_chain(
            anchor.descendant_chain_sha256.as_str(),
            suffix.sha256.as_str(),
        );
    }
    journal.stable = StableGeneration {
        generation: pending.generation,
        chain_sha256,
        position: verified_current.position,
    };
    journal.pending = None;
    write_journal(store, journal)?;
    Ok(io)
}

fn rollback_pending_suffix(
    store: &LocalThreadStore,
    journal: &mut AppendGenerationJournal,
    history_mode: ThreadHistoryMode,
    io: &mut AppendGenerationIo,
) -> ThreadStoreResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(journal.canonical_rollout_path.as_path())
        .map_err(source_error)?;
    file.set_len(journal.stable.position.end_byte_offset)
        .map_err(source_error)?;
    file.sync_all().map_err(source_error)?;
    let recovered = source_position(journal.canonical_rollout_path.as_path(), history_mode)?;
    io.add_source_bytes(recovered.source_bytes);
    if recovered.position.end_byte_offset != journal.stable.position.end_byte_offset
        || recovered.position.end_ordinal_exclusive != journal.stable.position.end_ordinal_exclusive
        || recovered.position.file_identity != journal.stable.position.file_identity
        || recovered.position.prefix_head_sha256 != journal.stable.position.prefix_head_sha256
        || recovered.position.prefix_middle_sha256 != journal.stable.position.prefix_middle_sha256
        || recovered.position.prefix_tail_sha256 != journal.stable.position.prefix_tail_sha256
    {
        return Err(invalid(
            "failed to restore the exact stable prefix after a torn append",
        ));
    }
    journal.stable.position = recovered.position;
    journal.pending = None;
    write_journal(store, journal)
}

fn pending_rollback_result(
    context: PendingRecoveryContext,
    io: AppendGenerationIo,
    reason: String,
) -> ThreadStoreResult<AppendGenerationIo> {
    tracing::warn!(reason, "rolled back an incomplete canonical rollout append");
    match context {
        PendingRecoveryContext::Continue => Ok(io),
        PendingRecoveryContext::FinishCurrentAppend => Err(ThreadStoreError::Internal {
            message: format!("canonical rollout append was rolled back: {reason}"),
        }),
    }
}

fn validate_journal_identity(
    journal: &AppendGenerationJournal,
    rollout_id: ThreadId,
    rollout_path: &Path,
) -> ThreadStoreResult<()> {
    let canonical_rollout_path = canonical_existing_path(rollout_path)?;
    if journal.version != APPEND_GENERATION_VERSION
        || journal.rollout_id != rollout_id
        || journal.canonical_rollout_path != canonical_rollout_path
    {
        return Err(invalid("rollout append generation identity mismatch"));
    }
    Ok(())
}

fn validate_current_position(
    journal: &AppendGenerationJournal,
) -> ThreadStoreResult<AppendGenerationIo> {
    let history_mode = if journal.stable.position.end_ordinal_exclusive.is_some() {
        ThreadHistoryMode::Paginated
    } else {
        ThreadHistoryMode::Legacy
    };
    let current = source_position(journal.canonical_rollout_path.as_path(), history_mode)?;
    // This is a generation-bound OS metadata fence, not a cryptographic proof of every source
    // byte. Canonical writes are serialized and use O_APPEND; an out-of-band mutation between
    // canonical operations changes the durable file identity/change marker and is rejected here.
    // Concurrent writers that can also forge OS metadata are outside this one-writer contract.
    if !positions_equal(&current.position, &journal.stable.position) {
        return Err(invalid(
            "source changed outside the canonical append-generation contract",
        ));
    }
    Ok(AppendGenerationIo {
        source_bytes: current.source_bytes,
        suffix_bytes: 0,
        suffix_items: 0,
    })
}

fn positions_equal(left: &SourcePosition, right: &SourcePosition) -> bool {
    left == right
}

struct SourceInspection {
    position: SourcePosition,
    source_bytes: u64,
}

fn source_position(
    path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<SourceInspection> {
    let before = std::fs::metadata(path).map_err(source_error)?;
    let before_generation = platform_file_generation(&before)?;
    let end_byte_offset = before.len();
    let (end_ordinal_exclusive, terminal_bytes) =
        terminal_ordinal_exclusive(path, end_byte_offset, history_mode)?;
    let (prefix_head_sha256, head_bytes) = hash_sample(path, 0, end_byte_offset)?;
    let (prefix_middle_sha256, middle_bytes) =
        hash_sample(path, middle_sample_start(end_byte_offset), end_byte_offset)?;
    let tail_start = end_byte_offset.saturating_sub(FENCE_SAMPLE_BYTES as u64);
    let (prefix_tail_sha256, tail_bytes) = hash_sample(path, tail_start, end_byte_offset)?;
    let after = std::fs::metadata(path).map_err(source_error)?;
    let after_generation = platform_file_generation(&after)?;
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || before_generation != after_generation
    {
        return Err(ThreadStoreError::Conflict {
            message: "rollout changed while its append generation was inspected".to_string(),
        });
    }
    let (file_identity, change_marker) = platform_file_generation(&after)?;
    Ok(SourceInspection {
        position: SourcePosition {
            end_byte_offset,
            end_ordinal_exclusive,
            modified_unix_nanos: modified_unix_nanos(after.modified().map_err(source_error)?)?,
            file_identity,
            change_marker,
            prefix_head_sha256,
            prefix_middle_sha256,
            prefix_tail_sha256,
        },
        source_bytes: terminal_bytes
            .saturating_add(head_bytes)
            .saturating_add(middle_bytes)
            .saturating_add(tail_bytes),
    })
}

fn validate_stored_prefix(
    path: &Path,
    stored: &SourcePosition,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<u64> {
    let before = std::fs::metadata(path).map_err(source_error)?;
    if before.len() < stored.end_byte_offset {
        return Err(invalid("pending append truncated the stored source prefix"));
    }
    let before_generation = platform_file_generation(&before)?;
    let file_identity = before_generation.0.as_str();
    if file_identity != stored.file_identity {
        return Err(invalid(
            "pending append replaced the stored source file identity",
        ));
    }
    // The stable metadata fence was checked immediately before begin_append. During the pending
    // generation the serialized canonical writer owns this path and appends only at EOF. These
    // bounded samples guard accidental prefix writes; they deliberately do not claim to prove an
    // arbitrary concurrent rewrite without rereading the complete prefix.
    let (prefix_head_sha256, head_bytes) = hash_sample(path, 0, stored.end_byte_offset)?;
    let (prefix_middle_sha256, middle_bytes) = hash_sample(
        path,
        middle_sample_start(stored.end_byte_offset),
        stored.end_byte_offset,
    )?;
    let tail_start = stored
        .end_byte_offset
        .saturating_sub(FENCE_SAMPLE_BYTES as u64);
    let (prefix_tail_sha256, tail_bytes) = hash_sample(path, tail_start, stored.end_byte_offset)?;
    let (end_ordinal_exclusive, terminal_bytes) =
        terminal_ordinal_exclusive(path, stored.end_byte_offset, history_mode)?;
    let after = std::fs::metadata(path).map_err(source_error)?;
    let after_generation = platform_file_generation(&after)?;
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || before_generation != after_generation
    {
        return Err(ThreadStoreError::Conflict {
            message: "rollout changed while its pending prefix was inspected".to_string(),
        });
    }
    if end_ordinal_exclusive != stored.end_ordinal_exclusive
        || prefix_head_sha256 != stored.prefix_head_sha256
        || prefix_middle_sha256 != stored.prefix_middle_sha256
        || prefix_tail_sha256 != stored.prefix_tail_sha256
    {
        return Err(invalid(
            "pending append changed the stored source prefix contract",
        ));
    }
    Ok(terminal_bytes
        .saturating_add(head_bytes)
        .saturating_add(middle_bytes)
        .saturating_add(tail_bytes))
}

fn terminal_ordinal_exclusive(
    path: &Path,
    end_byte_offset: u64,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<(Option<u64>, u64)> {
    match history_mode {
        ThreadHistoryMode::Legacy => Ok((None, 0)),
        ThreadHistoryMode::Paginated => {
            let file = File::open(path).map_err(source_error)?;
            let mut scanner = ReverseJsonlScanner::new_at(file, end_byte_offset)
                .map_err(source_error)?
                .with_strict_max_record_bytes(codex_rollout::MAX_ROLLOUT_LINE_BYTES);
            let line = match scanner.scan_next_rollout_line().map_err(source_error)? {
                Some(ScanOutcome::Parsed(line)) => line,
                Some(ScanOutcome::Rejected(err)) => {
                    return Err(invalid(format!("source terminal record is corrupt: {err}")));
                }
                None => return Err(invalid("source rollout is empty")),
            };
            let ordinal = line
                .ordinal
                .ok_or_else(|| invalid("paginated source terminal record has no ordinal"))?;
            let source_bytes = scanner.bytes_read();
            Ok((
                Some(
                    ordinal
                        .checked_add(1)
                        .ok_or_else(|| invalid("source ordinal overflow"))?,
                ),
                source_bytes,
            ))
        }
    }
}

fn middle_sample_start(end_byte_offset: u64) -> u64 {
    end_byte_offset.saturating_sub(FENCE_SAMPLE_BYTES as u64) / 2
}

struct SuffixSummary {
    item_count: u64,
    sha256: String,
    items_sha256: String,
    bytes_read: u64,
    end_ordinal_exclusive: Option<u64>,
}

fn summarize_suffix(
    path: &Path,
    start: u64,
    end: u64,
    start_ordinal_exclusive: Option<u64>,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<SuffixSummary> {
    let mut file = File::open(path).map_err(source_error)?;
    file.seek(SeekFrom::Start(start)).map_err(source_error)?;
    let mut reader = BufReader::new(file.take(end.saturating_sub(start)));
    let mut item_count = 0_u64;
    let mut next_ordinal = start_ordinal_exclusive;
    let mut hasher = Sha256::new();
    let mut items_hasher = Sha256::new();
    let mut bytes_read = 0_u64;
    loop {
        let mut line = Vec::new();
        let read = Read::by_ref(&mut reader)
            .take(codex_rollout::MAX_ROLLOUT_LINE_BYTES.saturating_add(1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(source_error)?;
        if read == 0 {
            break;
        }
        if read > codex_rollout::MAX_ROLLOUT_LINE_BYTES || !line.ends_with(b"\n") {
            return Err(invalid(
                "pending append contains an invalid rollout record boundary",
            ));
        }
        let value = serde_json::from_slice(line.as_slice())
            .map_err(|err| invalid(format!("pending append is corrupt: {err}")))?;
        let rollout_line = codex_rollout::decode_rollout_line(value)
            .map_err(|err| invalid(format!("pending append is invalid: {err}")))?;
        hash_item(&mut items_hasher, &rollout_line.item)?;
        if history_mode == ThreadHistoryMode::Paginated {
            let expected =
                next_ordinal.ok_or_else(|| invalid("missing paginated append ordinal"))?;
            if rollout_line.ordinal != Some(expected) {
                return Err(invalid(format!(
                    "pending append ordinal {:?} does not match expected {expected}",
                    rollout_line.ordinal
                )));
            }
            next_ordinal = Some(
                expected
                    .checked_add(1)
                    .ok_or_else(|| invalid("pending append ordinal overflow"))?,
            );
        }
        bytes_read = bytes_read.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        hasher.update(line);
        item_count = item_count.saturating_add(1);
    }
    Ok(SuffixSummary {
        item_count,
        sha256: format!("{:x}", hasher.finalize()),
        items_sha256: format!("{:x}", items_hasher.finalize()),
        bytes_read,
        end_ordinal_exclusive: next_ordinal,
    })
}

fn hash_items(items: &[RolloutItem]) -> ThreadStoreResult<String> {
    let mut hasher = Sha256::new();
    for item in items {
        hash_item(&mut hasher, item)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_item(hasher: &mut Sha256, item: &RolloutItem) -> ThreadStoreResult<()> {
    let bytes = serde_json::to_vec(item).map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to fingerprint canonical rollout append items: {err}"),
    })?;
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

fn load_journal(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
) -> ThreadStoreResult<Option<AppendGenerationJournal>> {
    let path = journal_path(store, rollout_id);
    let metadata = match std::fs::metadata(path.as_path()) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(invalid(format!("cannot stat append generation: {err}"))),
    };
    if metadata.len() > MAX_APPEND_GENERATION_BYTES {
        return Err(invalid("append generation exceeds its explicit size limit"));
    }
    let bytes = std::fs::read(path)
        .map_err(|err| invalid(format!("cannot read append generation: {err}")))?;
    let journal: AppendGenerationJournal = serde_json::from_slice(bytes.as_slice())
        .map_err(|err| invalid(format!("append generation is corrupt: {err}")))?;
    if journal.version != APPEND_GENERATION_VERSION {
        return Err(invalid(format!(
            "append generation version {} is unsupported",
            journal.version
        )));
    }
    Ok(Some(journal))
}

fn write_journal(
    store: &LocalThreadStore,
    journal: &AppendGenerationJournal,
) -> ThreadStoreResult<()> {
    let bytes = serde_json::to_vec(journal).map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to encode rollout append generation: {err}"),
    })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_APPEND_GENERATION_BYTES {
        return Err(invalid("append generation exceeds its explicit size limit"));
    }
    super::materialized_resume::atomic_write(
        journal_path(store, journal.rollout_id).as_path(),
        bytes.as_slice(),
    )
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to publish rollout append generation atomically: {err}"),
    })
}

pub(super) fn journal_path(store: &LocalThreadStore, rollout_id: ThreadId) -> PathBuf {
    store
        .config
        .codex_home
        .join(APPEND_GENERATION_DIRECTORY)
        .join(format!("{rollout_id}.json"))
}

fn materialized_generation(
    journal: &AppendGenerationJournal,
    checkpoint_anchor_id: Option<String>,
) -> MaterializedResumeAppendGeneration {
    MaterializedResumeAppendGeneration {
        generation_id: journal.generation_id.clone(),
        generation: journal.stable.generation,
        chain_sha256: journal.stable.chain_sha256.clone(),
        checkpoint_anchor_id,
    }
}

fn advance_chain(previous_chain_sha256: &str, suffix_sha256: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(previous_chain_sha256.as_bytes());
    hasher.update(suffix_sha256.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn genesis_chain(
    generation_id: &str,
    rollout_id: ThreadId,
    position: &SourcePosition,
) -> ThreadStoreResult<String> {
    let mut hasher = Sha256::new();
    hasher.update(generation_id.as_bytes());
    hasher.update(rollout_id.to_string().as_bytes());
    hasher.update(
        serde_json::to_vec(position).map_err(|err| ThreadStoreError::Internal {
            message: format!("failed to encode append-generation genesis: {err}"),
        })?,
    );
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_sample(path: &Path, start: u64, end: u64) -> ThreadStoreResult<(String, u64)> {
    let bounded_start = start.min(end);
    let sample_end = bounded_start
        .saturating_add(FENCE_SAMPLE_BYTES as u64)
        .min(end);
    let length = sample_end.saturating_sub(bounded_start);
    let mut file = File::open(path).map_err(source_error)?;
    file.seek(SeekFrom::Start(bounded_start))
        .map_err(source_error)?;
    let mut bytes = vec![
        0;
        usize::try_from(length).map_err(|_| invalid(
            "append-generation sample exceeds addressable memory"
        ))?
    ];
    file.read_exact(bytes.as_mut_slice())
        .map_err(source_error)?;
    Ok((format!("{:x}", Sha256::digest(bytes)), length))
}

fn canonical_existing_path(path: &Path) -> ThreadStoreResult<PathBuf> {
    std::fs::canonicalize(path).map_err(|err| ThreadStoreError::Internal {
        message: format!(
            "failed to canonicalize rollout source {}: {err}",
            path.display()
        ),
    })
}

fn modified_unix_nanos(modified: SystemTime) -> ThreadStoreResult<u64> {
    let nanos = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("source modified time predates the Unix epoch"))?
        .as_nanos();
    u64::try_from(nanos).map_err(|_| invalid("source modified time overflow"))
}

#[cfg(unix)]
pub(super) fn platform_file_generation(
    metadata: &std::fs::Metadata,
) -> ThreadStoreResult<PlatformFileGeneration> {
    use std::os::unix::fs::MetadataExt;
    Ok((
        format!("{}:{}", metadata.dev(), metadata.ino()),
        format!("{}:{}", metadata.ctime(), metadata.ctime_nsec()),
    ))
}

#[cfg(not(unix))]
pub(super) fn platform_file_generation(
    _metadata: &std::fs::Metadata,
) -> ThreadStoreResult<PlatformFileGeneration> {
    Err(invalid(
        "codex_resume_state_needs_platform_integrity: this platform has no supported append-generation OS metadata fence",
    ))
}

fn source_error(err: io::Error) -> ThreadStoreError {
    ThreadStoreError::Internal {
        message: format!("failed to inspect rollout append generation: {err}"),
    }
}

fn invalid(reason: impl std::fmt::Display) -> ThreadStoreError {
    super::materialized_resume::invalid_checkpoint(reason)
}
