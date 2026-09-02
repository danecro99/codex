use std::fs::File;
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
use codex_rollout::MaterializedResumeAppendGenerationLink;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::ScanOutcome;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

use super::LocalThreadStore;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

const APPEND_GENERATION_VERSION: u32 = 2;
const APPEND_GENERATION_DIRECTORY: &str = "rollout_append_generation_v1";
const MAX_APPEND_GENERATION_BYTES: u64 = 1024 * 1024;
const FENCE_SAMPLE_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AppendGenerationJournal {
    version: u32,
    rollout_id: ThreadId,
    canonical_rollout_path: PathBuf,
    generation_id: String,
    stable: StableGeneration,
    pending: Option<PendingAppend>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StableGeneration {
    generation: u64,
    chain_sha256: String,
    ancestry_base_generation: u64,
    ancestry_base_chain_sha256: String,
    ancestry: Vec<MaterializedResumeAppendGenerationLink>,
    position: SourcePosition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PendingAppend {
    generation: u64,
    expected_item_count: Option<u64>,
    history_mode: ThreadHistoryMode,
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
    expected_item_count: usize,
) -> ThreadStoreResult<bool> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        if super::materialized_resume::checkpoint_path(store, checkpoint_thread_id)
            .try_exists()
            .map_err(source_error)?
        {
            return Err(invalid(format!(
                "append generation is missing for rollout {rollout_id} while a materialized resume artifact exists"
            )));
        }
        return Ok(false);
    };
    recover_pending(store, &mut journal)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    begin_pending_append(
        store,
        &mut journal,
        history_mode,
        Some(u64::try_from(expected_item_count).unwrap_or(u64::MAX)),
    )
}

pub(super) fn begin_sync(
    store: &LocalThreadStore,
    checkpoint_thread_id: ThreadId,
    rollout_id: ThreadId,
    rollout_path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<bool> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        if super::materialized_resume::checkpoint_path(store, checkpoint_thread_id)
            .try_exists()
            .map_err(source_error)?
        {
            return Err(invalid(format!(
                "append generation is missing for rollout {rollout_id} while a materialized resume artifact exists"
            )));
        }
        return Ok(false);
    };
    recover_pending(store, &mut journal)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    begin_pending_append(store, &mut journal, history_mode, None)
}

fn begin_pending_append(
    store: &LocalThreadStore,
    journal: &mut AppendGenerationJournal,
    history_mode: ThreadHistoryMode,
    expected_item_count: Option<u64>,
) -> ThreadStoreResult<bool> {
    let generation = journal
        .stable
        .generation
        .checked_add(1)
        .ok_or_else(|| invalid("append generation overflow"))?;
    journal.pending = Some(PendingAppend {
        generation,
        expected_item_count,
        history_mode,
    });
    write_journal(store, journal)?;
    Ok(true)
}

pub(super) fn finish_append(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
) -> ThreadStoreResult<()> {
    let mut journal = load_journal(store, rollout_id)?
        .ok_or_else(|| invalid("append generation disappeared during canonical append"))?;
    recover_pending(store, &mut journal)
}

pub(super) fn load_current(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
    rollout_path: &Path,
) -> ThreadStoreResult<Option<MaterializedResumeAppendGeneration>> {
    let Some(mut journal) = load_journal(store, rollout_id)? else {
        return Ok(None);
    };
    recover_pending(store, &mut journal)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    Ok(Some(materialized_generation(&journal)))
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
    let position = source_position(canonical_rollout_path.as_path(), history_mode)?;
    let generation_id = ThreadId::new().to_string();
    let chain_sha256 = genesis_chain(&generation_id, rollout_id, &position)?;
    let journal = AppendGenerationJournal {
        version: APPEND_GENERATION_VERSION,
        rollout_id,
        canonical_rollout_path,
        generation_id,
        stable: StableGeneration {
            generation: 0,
            chain_sha256: chain_sha256.clone(),
            ancestry_base_generation: 0,
            ancestry_base_chain_sha256: chain_sha256,
            ancestry: Vec::new(),
            position,
        },
        pending: None,
    };
    write_journal(store, &journal)?;
    Ok(materialized_generation(&journal))
}

pub(super) fn validate_descendant(
    stored: &MaterializedResumeAppendGeneration,
    current: &MaterializedResumeAppendGeneration,
) -> ThreadStoreResult<()> {
    if stored.generation_id != current.generation_id || current.generation < stored.generation {
        return Err(invalid(
            "rollout append generation is not a descendant of the checkpoint",
        ));
    }
    if current.generation == stored.generation {
        return if current.chain_sha256 == stored.chain_sha256 {
            Ok(())
        } else {
            Err(invalid("rollout append generation chain was rewritten"))
        };
    }
    if current.ancestry_base_generation > stored.generation {
        return Err(invalid(
            "rollout append generation is missing the checkpoint ancestry anchor",
        ));
    }
    let expected_link_count = current
        .generation
        .checked_sub(current.ancestry_base_generation)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| invalid("rollout append generation ancestry length overflow"))?;
    if current.ancestry.len() != expected_link_count {
        return Err(invalid(format!(
            "rollout append generation ancestry has {} links; expected {expected_link_count}",
            current.ancestry.len()
        )));
    }
    let mut generation = current.ancestry_base_generation;
    let mut chain_sha256 = current.ancestry_base_chain_sha256.clone();
    let mut checkpoint_chain_verified = if generation == stored.generation {
        chain_sha256 == stored.chain_sha256
    } else {
        false
    };
    for link in &current.ancestry {
        generation = generation
            .checked_add(1)
            .ok_or_else(|| invalid("rollout append generation ancestry overflow"))?;
        if link.generation != generation {
            return Err(invalid(format!(
                "rollout append generation ancestry link {} does not match expected {generation}",
                link.generation
            )));
        }
        chain_sha256 = advance_chain(chain_sha256.as_str(), link.suffix_sha256.as_str());
        if generation == stored.generation {
            checkpoint_chain_verified = chain_sha256 == stored.chain_sha256;
        }
    }
    if !checkpoint_chain_verified {
        return Err(invalid(
            "rollout append generation ancestry does not reproduce the checkpoint chain",
        ));
    }
    if chain_sha256 != current.chain_sha256 {
        return Err(invalid(
            "rollout append generation ancestry does not produce the current chain",
        ));
    }
    Ok(())
}

pub(super) fn bind_checkpoint(
    store: &LocalThreadStore,
    rollout_id: ThreadId,
    rollout_path: &Path,
) -> ThreadStoreResult<()> {
    let mut journal = load_journal(store, rollout_id)?
        .ok_or_else(|| invalid("append generation disappeared after checkpoint publication"))?;
    recover_pending(store, &mut journal)?;
    validate_journal_identity(&journal, rollout_id, rollout_path)?;
    validate_current_position(&journal)?;
    journal.stable.ancestry_base_generation = journal.stable.generation;
    journal.stable.ancestry_base_chain_sha256 = journal.stable.chain_sha256.clone();
    journal.stable.ancestry.clear();
    write_journal(store, &journal)
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
) -> ThreadStoreResult<()> {
    let Some(pending) = journal.pending.clone() else {
        return Ok(());
    };
    let current = source_position(
        journal.canonical_rollout_path.as_path(),
        pending.history_mode,
    )?;
    if positions_equal(&current, &journal.stable.position) {
        journal.pending = None;
        write_journal(store, journal)?;
        return Ok(());
    }
    if current.end_byte_offset <= journal.stable.position.end_byte_offset {
        return Err(invalid(
            "source changed without completing its pending append",
        ));
    }
    validate_stored_prefix(
        journal.canonical_rollout_path.as_path(),
        &journal.stable.position,
        pending.history_mode,
    )?;
    let suffix = summarize_suffix(
        journal.canonical_rollout_path.as_path(),
        journal.stable.position.end_byte_offset,
        current.end_byte_offset,
        journal.stable.position.end_ordinal_exclusive,
        current.end_ordinal_exclusive,
        pending.history_mode,
    )?;
    let verified_current = source_position(
        journal.canonical_rollout_path.as_path(),
        pending.history_mode,
    )?;
    if !positions_equal(&current, &verified_current) {
        return Err(ThreadStoreError::Conflict {
            message: "rollout changed while its pending suffix was inspected".to_string(),
        });
    }
    if let Some(expected_item_count) = pending.expected_item_count
        && suffix.item_count != expected_item_count
    {
        return Err(invalid(format!(
            "pending append wrote {} items; expected {}",
            suffix.item_count, expected_item_count
        )));
    }
    let chain_sha256 = advance_chain(journal.stable.chain_sha256.as_str(), suffix.sha256.as_str());
    journal
        .stable
        .ancestry
        .push(MaterializedResumeAppendGenerationLink {
            generation: pending.generation,
            suffix_sha256: suffix.sha256,
        });
    let ancestry_base_generation = journal.stable.ancestry_base_generation;
    let ancestry_base_chain_sha256 = journal.stable.ancestry_base_chain_sha256.clone();
    let ancestry = std::mem::take(&mut journal.stable.ancestry);
    journal.stable = StableGeneration {
        generation: pending.generation,
        chain_sha256,
        ancestry_base_generation,
        ancestry_base_chain_sha256,
        ancestry,
        position: current,
    };
    journal.pending = None;
    write_journal(store, journal)
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

fn validate_current_position(journal: &AppendGenerationJournal) -> ThreadStoreResult<()> {
    let history_mode = if journal.stable.position.end_ordinal_exclusive.is_some() {
        ThreadHistoryMode::Paginated
    } else {
        ThreadHistoryMode::Legacy
    };
    let current = source_position(journal.canonical_rollout_path.as_path(), history_mode)?;
    if !positions_equal(&current, &journal.stable.position) {
        return Err(invalid(
            "source changed outside the canonical append-generation contract",
        ));
    }
    Ok(())
}

fn positions_equal(left: &SourcePosition, right: &SourcePosition) -> bool {
    left == right
}

fn source_position(
    path: &Path,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<SourcePosition> {
    let before = std::fs::metadata(path).map_err(source_error)?;
    let (before_file_identity, _) = platform_file_generation(&before);
    let end_byte_offset = before.len();
    let end_ordinal_exclusive = terminal_ordinal_exclusive(path, end_byte_offset, history_mode)?;
    let (prefix_head_sha256, _) = hash_sample(path, 0, end_byte_offset)?;
    let (prefix_middle_sha256, _) =
        hash_sample(path, middle_sample_start(end_byte_offset), end_byte_offset)?;
    let tail_start = end_byte_offset.saturating_sub(FENCE_SAMPLE_BYTES as u64);
    let (prefix_tail_sha256, _) = hash_sample(path, tail_start, end_byte_offset)?;
    let after = std::fs::metadata(path).map_err(source_error)?;
    let (after_file_identity, _) = platform_file_generation(&after);
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || before_file_identity != after_file_identity
    {
        return Err(ThreadStoreError::Conflict {
            message: "rollout changed while its append generation was inspected".to_string(),
        });
    }
    let (file_identity, change_marker) = platform_file_generation(&after);
    Ok(SourcePosition {
        end_byte_offset,
        end_ordinal_exclusive,
        modified_unix_nanos: modified_unix_nanos(after.modified().map_err(source_error)?)?,
        file_identity,
        change_marker,
        prefix_head_sha256,
        prefix_middle_sha256,
        prefix_tail_sha256,
    })
}

fn validate_stored_prefix(
    path: &Path,
    stored: &SourcePosition,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<()> {
    let before = std::fs::metadata(path).map_err(source_error)?;
    if before.len() < stored.end_byte_offset {
        return Err(invalid("pending append truncated the stored source prefix"));
    }
    let (file_identity, _) = platform_file_generation(&before);
    if file_identity != stored.file_identity {
        return Err(invalid(
            "pending append replaced the stored source file identity",
        ));
    }
    let (prefix_head_sha256, _) = hash_sample(path, 0, stored.end_byte_offset)?;
    let (prefix_middle_sha256, _) = hash_sample(
        path,
        middle_sample_start(stored.end_byte_offset),
        stored.end_byte_offset,
    )?;
    let tail_start = stored
        .end_byte_offset
        .saturating_sub(FENCE_SAMPLE_BYTES as u64);
    let (prefix_tail_sha256, _) = hash_sample(path, tail_start, stored.end_byte_offset)?;
    let end_ordinal_exclusive =
        terminal_ordinal_exclusive(path, stored.end_byte_offset, history_mode)?;
    let after = std::fs::metadata(path).map_err(source_error)?;
    let (after_file_identity, _) = platform_file_generation(&after);
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || file_identity != after_file_identity
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
    Ok(())
}

fn terminal_ordinal_exclusive(
    path: &Path,
    end_byte_offset: u64,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<Option<u64>> {
    match history_mode {
        ThreadHistoryMode::Legacy => Ok(None),
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
            Ok(Some(
                ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid("source ordinal overflow"))?,
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
}

fn summarize_suffix(
    path: &Path,
    start: u64,
    end: u64,
    start_ordinal_exclusive: Option<u64>,
    end_ordinal_exclusive: Option<u64>,
    history_mode: ThreadHistoryMode,
) -> ThreadStoreResult<SuffixSummary> {
    let mut file = File::open(path).map_err(source_error)?;
    file.seek(SeekFrom::Start(start)).map_err(source_error)?;
    let mut reader = BufReader::new(file.take(end.saturating_sub(start)));
    let mut item_count = 0_u64;
    let mut next_ordinal = start_ordinal_exclusive;
    let mut hasher = Sha256::new();
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
        hasher.update(line);
        item_count = item_count.saturating_add(1);
    }
    if next_ordinal != end_ordinal_exclusive {
        return Err(invalid("pending append terminal ordinal mismatch"));
    }
    Ok(SuffixSummary {
        item_count,
        sha256: format!("{:x}", hasher.finalize()),
    })
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
) -> MaterializedResumeAppendGeneration {
    MaterializedResumeAppendGeneration {
        generation_id: journal.generation_id.clone(),
        generation: journal.stable.generation,
        chain_sha256: journal.stable.chain_sha256.clone(),
        ancestry_base_generation: journal.stable.ancestry_base_generation,
        ancestry_base_chain_sha256: journal.stable.ancestry_base_chain_sha256.clone(),
        ancestry: journal.stable.ancestry.clone(),
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
fn platform_file_generation(metadata: &std::fs::Metadata) -> (String, String) {
    use std::os::unix::fs::MetadataExt;
    (
        format!("{}:{}", metadata.dev(), metadata.ino()),
        format!("{}:{}", metadata.ctime(), metadata.ctime_nsec()),
    )
}

#[cfg(windows)]
fn platform_file_generation(metadata: &std::fs::Metadata) -> (String, String) {
    use std::os::windows::fs::MetadataExt;
    (
        format!("creation:{}", metadata.creation_time()),
        format!("last_write:{}", metadata.last_write_time()),
    )
}

#[cfg(not(any(unix, windows)))]
fn platform_file_generation(metadata: &std::fs::Metadata) -> (String, String) {
    (format!("len:{}", metadata.len()), String::new())
}

fn source_error(err: io::Error) -> ThreadStoreError {
    ThreadStoreError::Internal {
        message: format!("failed to inspect rollout append generation: {err}"),
    }
}

fn invalid(reason: impl std::fmt::Display) -> ThreadStoreError {
    super::materialized_resume::invalid_checkpoint(reason)
}
