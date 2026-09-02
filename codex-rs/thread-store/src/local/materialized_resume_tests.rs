use std::fs::File;
use std::fs::OpenOptions;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::Arc;

use chrono::Utc;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::protocol::TruncationPolicy;
use codex_rollout::MATERIALIZED_RESUME_STATE_VERSION;
use codex_rollout::MaterializedAutoCompactWindow;
use codex_rollout::MaterializedResumeState;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutLine;
use pretty_assertions::assert_eq;
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::ArchiveThreadParams;
use crate::DeleteThreadParams;
use crate::LoadModelContextParams;
use crate::ResumeCheckpointOutcome;
use crate::ThreadStore;
use crate::local::model_context::load_latest_model_context;
use crate::local::test_support::test_config;
use crate::local::test_support::write_session_file_with_history_mode;

fn user_message(text: String) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn state() -> MaterializedResumeState {
    let window_id = Uuid::now_v7().to_string();
    MaterializedResumeState {
        version: MATERIALIZED_RESUME_STATE_VERSION,
        history: Arc::new(Vec::new()),
        previous_turn_settings: None,
        reference_context_item: None,
        world_state_baseline: None,
        mcp_resource_origins: None,
        auto_compact_window: MaterializedAutoCompactWindow {
            window_number: 0,
            first_window_id: window_id.clone(),
            previous_window_id: None,
            window_id,
        },
        token_info: None,
        last_agent_status: None,
        truncation_policy: TruncationPolicy::Tokens(128_000),
        auto_compact_window_prefill_input_tokens: None,
        has_prior_user_turns: false,
    }
}

fn write_large_legacy_rollout(home: &std::path::Path, uuid: Uuid) -> std::path::PathBuf {
    let path = write_session_file_with_history_mode(
        home,
        "2025-01-03T15-00-00",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write session file");
    let mut items = Vec::new();
    for index in 0..2_048 {
        items.push(user_message(format!("{index}:{}", "x".repeat(2_048))));
    }
    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open large rollout");
    for item in items {
        let line = RolloutLine {
            timestamp: "2025-01-03T15:00:00Z".to_string(),
            ordinal: None,
            item,
        };
        serde_json::to_writer(&mut file, &line).expect("encode large rollout item");
        file.write_all(b"\n").expect("terminate rollout item");
    }
    file.sync_all().expect("sync large rollout");
    path
}

fn rewrite_payload_byte_at_or_after(path: &std::path::Path, search_start: usize) -> usize {
    let transcript = std::fs::read(path).expect("read large transcript");
    let relative = transcript[search_start..]
        .iter()
        .position(|byte| *byte == b'x')
        .expect("payload byte after requested offset");
    let offset = search_start + relative;
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open source without truncation");
    file.seek(SeekFrom::Start(
        u64::try_from(offset).expect("payload offset"),
    ))
    .expect("seek payload byte");
    file.write_all(b"y").expect("rewrite payload byte");
    file.sync_all().expect("sync payload rewrite");
    offset
}

fn rewrite_middle_payload_byte(path: &std::path::Path) {
    let length = usize::try_from(std::fs::metadata(path).expect("source metadata").len())
        .expect("source length");
    rewrite_payload_byte_at_or_after(path, length / 2);
}

fn rewrite_unsampled_payload_byte(path: &std::path::Path) {
    let length = usize::try_from(std::fs::metadata(path).expect("source metadata").len())
        .expect("source length");
    let offset = rewrite_payload_byte_at_or_after(path, length / 4);
    let middle_start = length.saturating_sub(FENCE_SAMPLE_BYTES) / 2;
    assert!(offset >= FENCE_SAMPLE_BYTES);
    assert!(offset < middle_start || offset >= middle_start + FENCE_SAMPLE_BYTES);
    assert!(offset < length.saturating_sub(FENCE_SAMPLE_BYTES));
}

async fn publish_loaded_state(
    store: &LocalThreadStore,
    thread_id: ThreadId,
    loaded: &crate::StoredModelContext,
) {
    let source = loaded
        .materialized_resume
        .as_ref()
        .expect("materialization fence")
        .source
        .clone();
    store
        .publish_materialized_resume_state(PublishMaterializedResumeParams {
            thread_id,
            fence: MaterializedResumePublicationFence::Loaded(Box::new(source)),
            state: state(),
            max_state_bytes: 64 * 1024 * 1024,
        })
        .await
        .expect("publish materialized state");
}

async fn append_with_generation(
    store: &LocalThreadStore,
    thread_id: ThreadId,
    path: &std::path::Path,
    history_mode: ThreadHistoryMode,
    item: &RolloutItem,
) {
    let started = crate::local::append_generation::begin_append(
        store,
        thread_id,
        thread_id,
        path,
        history_mode,
        1,
    )
    .expect("begin canonical append");
    assert!(
        started,
        "published checkpoint must own an append generation"
    );
    codex_rollout::append_rollout_item_to_path(path, item)
        .await
        .expect("append rollout item");
    crate::local::append_generation::finish_append(store, thread_id)
        .expect("finish canonical append");
}

#[tokio::test]
async fn second_unchanged_resume_reads_only_bounded_checkpoint_input() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_001);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_large_legacy_rollout(home.path(), uuid);
    let transcript_before = std::fs::read(&path).expect("read transcript");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first resume");
    assert_eq!(first.diagnostics.outcome, ResumeCheckpointOutcome::Miss);
    assert!(first.diagnostics.source_items > 2_000);
    assert!(first.diagnostics.source_bytes >= transcript_before.len() as u64);
    publish_loaded_state(&store, thread_id, &first).await;

    let second = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("second resume");
    assert_eq!(second.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(second.diagnostics.source_items, 0);
    assert_eq!(second.diagnostics.suffix_items, 0);
    assert!(second.diagnostics.source_bytes <= 4 * FENCE_SAMPLE_BYTES as u64);
    assert_eq!(second.items.len(), 1);
    assert_eq!(
        std::fs::read(path).expect("read preserved transcript"),
        transcript_before
    );

    let replay = store
        .load_latest_model_context_for_replay(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("source replay must not consume private resume state");
    assert_eq!(replay.diagnostics.outcome, ResumeCheckpointOutcome::Miss);
    assert!(replay.diagnostics.source_items > 2_000);
    assert!(replay.items.len() > 2_000);
    assert_eq!(replay.materialized_resume, None);
}

#[tokio::test]
async fn append_after_checkpoint_reads_and_republishes_only_the_suffix() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_002);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-01",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first resume");
    publish_loaded_state(&store, thread_id, &first).await;

    let appended = user_message("suffix".to_string());
    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &appended,
    )
    .await;
    let suffix = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("suffix resume");
    assert_eq!(suffix.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(suffix.diagnostics.source_items, 1);
    assert_eq!(suffix.diagnostics.suffix_items, 1);
    assert!(suffix.diagnostics.suffix_bytes < std::fs::metadata(&path).unwrap().len());
    assert_eq!(
        serde_json::to_value(&suffix.items[1]).expect("serialize suffix"),
        serde_json::to_value(appended).expect("serialize expected suffix")
    );
    publish_loaded_state(&store, thread_id, &suffix).await;
    let artifact_directory = checkpoint_path(&store, thread_id)
        .parent()
        .expect("artifact directory")
        .to_path_buf();
    let artifact_names = std::fs::read_dir(artifact_directory)
        .expect("read artifact directory")
        .map(|entry| entry.expect("artifact entry").file_name())
        .collect::<Vec<_>>();
    assert_eq!(
        artifact_names,
        vec![std::ffi::OsString::from(format!("{thread_id}.json"))]
    );

    let unchanged = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect("unchanged extended resume");
    assert_eq!(unchanged.diagnostics.suffix_items, 0);
    assert_eq!(unchanged.items.len(), 1);
}

#[tokio::test]
async fn append_ancestry_verifies_a_checkpoint_published_before_journal_rebinding() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_012);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-12",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first resume");
    publish_loaded_state(&store, thread_id, &first).await;

    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &user_message("first suffix".to_string()),
    )
    .await;
    let after_first_append = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first suffix resume");
    let source = after_first_append
        .materialized_resume
        .expect("materialized resume")
        .source;
    let artifact = codex_rollout::MaterializedResume {
        source,
        state: Some(state()),
        max_state_bytes: Some(64 * 1024 * 1024),
    };
    atomic_write(
        checkpoint_path(&store, thread_id).as_path(),
        serde_json::to_vec(&artifact)
            .expect("encode checkpoint")
            .as_slice(),
    )
    .expect("publish checkpoint without journal rebinding");

    let second_suffix = user_message("second suffix".to_string());
    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &second_suffix,
    )
    .await;
    let resumed = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect("ancestry must reproduce the published checkpoint chain");
    assert_eq!(resumed.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(resumed.diagnostics.suffix_items, 1);
    assert_eq!(
        serde_json::to_value(&resumed.items[1]).expect("serialize resumed suffix"),
        serde_json::to_value(second_suffix).expect("serialize expected suffix")
    );
}

#[tokio::test]
async fn paginated_checkpoint_follows_a_normal_descendant_segment() {
    let home = TempDir::new().expect("temp dir");
    let config = test_config(home.path());
    let runtime = codex_state::StateRuntime::init(
        config.sqlite.clone(),
        config.default_model_provider_id.clone(),
    )
    .await
    .expect("state runtime");
    let uuid = Uuid::from_u128(/*v*/ 4_008);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let root_path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-08",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write root segment");
    let mut root_metadata = codex_state::ThreadMetadataBuilder::new(
        thread_id,
        root_path.clone(),
        Utc::now(),
        SessionSource::Cli,
    );
    root_metadata.history_mode = ThreadHistoryMode::Paginated;
    runtime
        .upsert_thread(&root_metadata.build(config.default_model_provider_id.as_str()))
        .await
        .expect("seed root metadata");
    let store = LocalThreadStore::new(config.clone(), Some(runtime.clone()));
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(root_path.clone()),
        },
    )
    .await
    .expect("load root segment");
    publish_loaded_state(&store, thread_id, &first).await;

    let root_end = std::fs::metadata(root_path.as_path())
        .expect("root metadata")
        .len();
    let history_base = HistoryPosition {
        thread_id,
        end_ordinal_exclusive: 1,
        end_byte_offset: root_end,
    };
    let descendant_rollout_id = ThreadId::new();
    let descendant_path = root_path.with_file_name(format!(
        "rollout-2025-01-03T15-00-09-{descendant_rollout_id}.jsonl"
    ));
    let mut descendant = File::create(descendant_path.as_path()).expect("create descendant");
    for line in [
        RolloutLine {
            timestamp: "2025-01-03T15:00:09Z".to_string(),
            ordinal: Some(1),
            item: RolloutItem::SessionMeta(SessionMetaLine {
                meta: SessionMeta {
                    session_id: thread_id.into(),
                    id: thread_id,
                    history_mode: ThreadHistoryMode::Paginated,
                    history_base: Some(history_base),
                    ..SessionMeta::default()
                },
                git: None,
            }),
        },
        RolloutLine {
            timestamp: "2025-01-03T15:00:09Z".to_string(),
            ordinal: Some(2),
            item: user_message("descendant suffix".to_string()),
        },
    ] {
        serde_json::to_writer(&mut descendant, &line).expect("encode descendant line");
        descendant
            .write_all(b"\n")
            .expect("terminate descendant line");
    }
    descendant.sync_all().expect("sync descendant");
    let mut descendant_metadata = codex_state::ThreadMetadataBuilder::new(
        thread_id,
        descendant_path.clone(),
        Utc::now(),
        SessionSource::Cli,
    );
    descendant_metadata.history_mode = ThreadHistoryMode::Paginated;
    runtime
        .upsert_thread(&descendant_metadata.build(config.default_model_provider_id.as_str()))
        .await
        .expect("advance current rollout path");

    let extended = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(descendant_path.clone()),
        },
    )
    .await
    .expect("checkpoint must follow descendant segment");
    assert_eq!(extended.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(extended.diagnostics.suffix_items, 1);
    assert_eq!(
        serde_json::to_value(&extended.items[1]).expect("serialize descendant suffix"),
        serde_json::to_value(user_message("descendant suffix".to_string()))
            .expect("serialize expected suffix")
    );
    publish_loaded_state(&store, thread_id, &extended).await;

    let unchanged = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(descendant_path),
        },
    )
    .await
    .expect("unchanged descendant resume");
    assert_eq!(unchanged.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(unchanged.diagnostics.suffix_items, 0);
}

#[tokio::test]
async fn append_generation_rejects_sampled_middle_rewrite() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_009);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_large_legacy_rollout(home.path(), uuid);
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load large source");
    publish_loaded_state(&store, thread_id, &first).await;

    rewrite_middle_payload_byte(path.as_path());
    codex_rollout::append_rollout_item_to_path(&path, &user_message("raw append".to_string()))
        .await
        .expect("append outside canonical writer");

    let error = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect_err("out-of-contract middle rewrite must fail");
    assert!(
        error
            .to_string()
            .contains("outside the canonical append-generation contract"),
        "{error}"
    );
}

#[tokio::test]
async fn pending_append_rejects_unsampled_prefix_rewrite_before_suffix_commit() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_010);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_large_legacy_rollout(home.path(), uuid);
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load large source");
    publish_loaded_state(&store, thread_id, &first).await;

    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Legacy,
            1,
        )
        .expect("begin pending append")
    );
    rewrite_unsampled_payload_byte(path.as_path());
    codex_rollout::append_rollout_item_to_path(
        path.as_path(),
        &user_message("legitimate suffix".to_string()),
    )
    .await
    .expect("append legitimate suffix");

    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("rewritten pending prefix must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("complete stored source prefix digest"),
        "{error}"
    );
}

#[tokio::test]
async fn forged_higher_generation_without_checkpoint_ancestry_anchor_is_loud() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_011);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-11",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load source");
    publish_loaded_state(&store, thread_id, &first).await;

    let journal_path = crate::local::append_generation::journal_path(&store, thread_id);
    let mut journal: serde_json::Value = serde_json::from_slice(
        std::fs::read(journal_path.as_path())
            .expect("read append generation")
            .as_slice(),
    )
    .expect("decode append generation");
    let stable = journal
        .get_mut("stable")
        .and_then(serde_json::Value::as_object_mut)
        .expect("stable generation");
    stable.insert("generation".to_string(), serde_json::json!(1));
    stable.insert(
        "chain_sha256".to_string(),
        serde_json::json!("11".repeat(32)),
    );
    stable.insert("ancestry_base_generation".to_string(), serde_json::json!(1));
    stable.insert(
        "ancestry_base_chain_sha256".to_string(),
        serde_json::json!("22".repeat(32)),
    );
    stable.insert("ancestry".to_string(), serde_json::json!([]));
    std::fs::write(
        journal_path,
        serde_json::to_vec(&journal).expect("encode forged append generation"),
    )
    .expect("write forged append generation");

    let error = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect_err("higher generation without ancestry anchor must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
    assert!(error.to_string().contains("ancestry anchor"), "{error}");
}

#[tokio::test]
async fn paginated_suffix_ordinal_gap_is_loud() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_007);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-06",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first resume");
    publish_loaded_state(&store, thread_id, &first).await;

    let unchanged = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("unchanged paginated resume");
    assert_eq!(unchanged.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(unchanged.diagnostics.source_items, 1);
    assert_eq!(unchanged.diagnostics.suffix_items, 0);
    assert!(unchanged.diagnostics.source_bytes <= 5 * FENCE_SAMPLE_BYTES as u64);

    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open paginated rollout");
    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Paginated,
            1,
        )
        .expect("begin malformed canonical append")
    );
    serde_json::to_writer(
        &mut file,
        &RolloutLine {
            timestamp: "2025-01-03T15:00:07Z".to_string(),
            ordinal: Some(3),
            item: user_message("ordinal gap".to_string()),
        },
    )
    .expect("encode ordinal gap");
    file.write_all(b"\n").expect("terminate ordinal gap");
    file.sync_all().expect("sync ordinal gap");

    let error = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect_err("ordinal gap must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction")
    );
    assert!(error.to_string().contains("ordinal"), "{error}");
}

#[tokio::test]
async fn invalid_artifact_and_rewritten_source_fence_are_loud() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_003);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-02",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write session file");
    codex_rollout::append_rollout_item_to_path(&path, &user_message("original".to_string()))
        .await
        .expect("append original");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let first = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("first resume");
    publish_loaded_state(&store, thread_id, &first).await;

    let artifact_path = checkpoint_path(&store, thread_id);
    let valid_artifact = std::fs::read(&artifact_path).expect("read valid artifact");
    let mut artifact: codex_rollout::MaterializedResume =
        serde_json::from_slice(&std::fs::read(&artifact_path).expect("read artifact"))
            .expect("decode artifact");
    artifact.state.as_mut().expect("state").version =
        MATERIALIZED_RESUME_STATE_VERSION.saturating_add(1);
    std::fs::write(
        &artifact_path,
        serde_json::to_vec(&artifact).expect("encode version mismatch"),
    )
    .expect("write version mismatch");
    let version = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect_err("version mismatch must fail");
    assert!(version.to_string().contains("version"), "{version}");

    std::fs::write(&artifact_path, valid_artifact.as_slice()).expect("restore valid artifact");
    let mut artifact: codex_rollout::MaterializedResume =
        serde_json::from_slice(&std::fs::read(&artifact_path).expect("read artifact"))
            .expect("decode artifact");
    artifact.source.rollout_id =
        ThreadId::from_string(&Uuid::from_u128(/*v*/ 4_999).to_string()).expect("thread id");
    std::fs::write(
        &artifact_path,
        serde_json::to_vec(&artifact).expect("encode identity mismatch"),
    )
    .expect("write identity mismatch");
    let identity = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect_err("source identity mismatch must fail");
    assert!(
        identity
            .to_string()
            .contains("codex_resume_state_needs_compaction")
    );

    std::fs::write(&artifact_path, valid_artifact.as_slice()).expect("restore valid artifact");
    std::fs::write(checkpoint_path(&store, thread_id), b"not-json").expect("corrupt artifact");
    let corrupt = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect_err("corrupt artifact must fail");
    assert!(
        corrupt
            .to_string()
            .contains("codex_resume_state_needs_compaction")
    );

    std::fs::write(&artifact_path, valid_artifact.as_slice()).expect("restore valid artifact");
    let mut transcript = std::fs::read(&path).expect("read transcript");
    let position = transcript
        .windows("original".len())
        .position(|window| window == b"original")
        .expect("original text");
    transcript[position..position + "original".len()].copy_from_slice(b"rewritte");
    std::fs::write(&path, transcript).expect("rewrite source prefix");
    let rewritten = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect_err("rewritten source must fail");
    assert!(
        rewritten
            .to_string()
            .contains("codex_resume_state_needs_compaction")
    );
}

#[tokio::test]
async fn archive_and_delete_remove_only_the_owned_artifact() {
    let home = TempDir::new().expect("temp dir");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let unrelated_thread_id =
        ThreadId::from_string(&Uuid::from_u128(/*v*/ 4_006).to_string()).expect("thread id");
    let unrelated_artifact = checkpoint_path(&store, unrelated_thread_id);
    let unrelated_generation =
        crate::local::append_generation::journal_path(&store, unrelated_thread_id);
    std::fs::create_dir_all(unrelated_artifact.parent().expect("artifact directory"))
        .expect("create artifact directory");
    std::fs::write(&unrelated_artifact, b"unrelated checkpoint")
        .expect("write unrelated checkpoint");
    std::fs::create_dir_all(unrelated_generation.parent().expect("generation directory"))
        .expect("create generation directory");
    std::fs::write(&unrelated_generation, b"unrelated generation")
        .expect("write unrelated generation");
    for (index, archive) in [(4_004_u128, true), (4_005_u128, false)] {
        let uuid = Uuid::from_u128(index);
        let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
        let path = write_session_file_with_history_mode(
            home.path(),
            if archive {
                "2025-01-03T15-00-03"
            } else {
                "2025-01-03T15-00-04"
            },
            uuid,
            ThreadHistoryMode::Legacy,
        )
        .expect("write session file");
        let loaded = load_latest_model_context(
            &store,
            LoadModelContextParams {
                thread_id,
                include_archived: false,
                rollout_path: Some(path),
            },
        )
        .await
        .expect("load source");
        publish_loaded_state(&store, thread_id, &loaded).await;
        let artifact = checkpoint_path(&store, thread_id);
        let generation = crate::local::append_generation::journal_path(&store, thread_id);
        assert!(artifact.exists());
        assert!(generation.exists());
        if archive {
            store
                .archive_thread(ArchiveThreadParams { thread_id })
                .await
                .expect("archive thread");
        } else {
            store
                .delete_thread(DeleteThreadParams { thread_id })
                .await
                .expect("delete thread");
        }
        assert!(!artifact.exists());
        assert!(!generation.exists());
        assert_eq!(
            std::fs::read(&unrelated_artifact).expect("read unrelated checkpoint"),
            b"unrelated checkpoint"
        );
        assert_eq!(
            std::fs::read(&unrelated_generation).expect("read unrelated generation"),
            b"unrelated generation"
        );
    }
}
