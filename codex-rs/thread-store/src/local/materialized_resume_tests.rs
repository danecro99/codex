use std::fs::File;
use std::fs::OpenOptions;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::Arc;

use chrono::Utc;
use codex_protocol::ResponseItemId;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::protocol::ThreadMemoryMode;
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
use crate::AppendThreadItemsParams;
use crate::CreateThreadParams;
use crate::ArchiveThreadParams;
use crate::DeleteThreadParams;
use crate::LoadModelContextParams;
use crate::ResumeCheckpointOutcome;
use crate::ResumeThreadParams;
use crate::ThreadPersistenceMetadata;
use crate::ThreadStore;
use crate::ThreadStoreError;
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

fn assistant_message(text: String) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn reasoning(content: Option<Vec<ReasoningItemContent>>) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Reasoning {
            id: Some(ResponseItemId::with_suffix("rs", "1")),
            summary: vec![ReasoningItemReasoningSummary::SummaryText {
                text: "summary".to_string(),
            }],
            content,
            encrypted_content: Some("encrypted".to_string()),
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn function_call(call_id: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            namespace: None,
            arguments: r#"{"command":["echo","hi"]}"#.to_string(),
            encrypted_function_args: None,
            call_id: call_id.to_string(),
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn function_call_output(call_id: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some(call_id.to_string()),
            name: None,
            namespace: None,
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("hi\n".to_string()),
                success: Some(true),
            },
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn message_with_host_metadata(cell_id: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "same visible text".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: Some(
                InternalChatMessageMetadataPassthrough {
                    cell_id: Some(cell_id.to_string()),
                    ..InternalChatMessageMetadataPassthrough::default()
                },
            ),
        }
        .into(),
    )
}

/// What resume must hand back for `item`: exactly what its canonical record decodes to.
fn durable_readback_form(item: &RolloutItem) -> RolloutItem {
    codex_rollout::decode_rollout_line(
        serde_json::to_value(RolloutLine {
            timestamp: "2025-01-03T15:00:19Z".to_string(),
            ordinal: Some(1),
            item: item.clone(),
        })
        .expect("encode canonical record"),
    )
    .expect("decode canonical record")
    .item
}

fn state() -> MaterializedResumeState {
    let window_id = Uuid::now_v7().to_string();
    MaterializedResumeState {
        version: MATERIALIZED_RESUME_STATE_VERSION,
        materialized_model: "test-model".to_string(),
        history: Arc::new(Vec::new()),
        guardian_history: None,
        previous_turn_settings: None,
        reference_context_item: None,
        world_state_baseline: None,
        mcp_resource_origins: None,
        owned_startup_cwd: None,
        auto_compact_window: MaterializedAutoCompactWindow {
            window_number: 0,
            first_window_id: window_id.clone(),
            previous_window_id: None,
            window_id,
        },
        token_info: None,
        latest_token_usage_record: None,
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
) -> (
    crate::local::append_generation::AppendGenerationIo,
    crate::local::append_generation::AppendGenerationIo,
) {
    let start = crate::local::append_generation::begin_append(
        store,
        thread_id,
        thread_id,
        path,
        history_mode,
        std::slice::from_ref(item),
    )
    .expect("begin canonical append");
    assert!(
        start.started,
        "published checkpoint must own an append generation"
    );
    codex_rollout::append_rollout_item_to_path(path, item)
        .await
        .expect("append rollout item");
    let finish_io = crate::local::append_generation::finish_append(store, thread_id)
        .expect("finish canonical append");
    (start.io, finish_io)
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
    assert!(second.diagnostics.source_bytes <= 5 * FENCE_SAMPLE_BYTES as u64);
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

/// A turn's real item shapes must survive the canonical append and come back on resume.
///
/// The write intent is fingerprinted before the writer runs and re-checked against the decoded
/// durable records, so any item whose persisted encoding differs from its in-memory form used to
/// be rejected as a foreign write and truncated back off the transcript.
#[tokio::test]
async fn canonical_append_keeps_every_turn_item_shape_durable() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_019);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-19",
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

    let turn = vec![
        user_message("what changed?".to_string()),
        reasoning(Some(vec![ReasoningItemContent::Text {
            text: "raw reasoning".to_string(),
        }])),
        reasoning(Some(Vec::new())),
        reasoning(Some(vec![ReasoningItemContent::ReasoningText {
            text: "visible reasoning".to_string(),
        }])),
        function_call("call-1"),
        function_call_output("call-1"),
        assistant_message("nothing changed".to_string()),
    ];
    for item in &turn {
        append_with_generation(
            &store,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Paginated,
            item,
        )
        .await;
    }

    let resumed = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("resume after appending a full turn");
    assert_eq!(resumed.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(
        resumed.diagnostics.suffix_items,
        u64::try_from(turn.len()).expect("suffix items")
    );
    let durable = resumed
        .items
        .iter()
        .skip(1)
        .map(|item| serde_json::to_value(item).expect("serialize durable item"))
        .collect::<Vec<_>>();
    let expected = turn
        .iter()
        .map(|item| {
            serde_json::to_value(durable_readback_form(item)).expect("serialize expected item")
        })
        .collect::<Vec<_>>();
    assert_eq!(durable, expected);

    // A resumed thread must be able to record a new turn and read that back too.
    publish_loaded_state(&store, thread_id, &resumed).await;
    let next_turn = vec![
        user_message("and now?".to_string()),
        reasoning(Some(vec![ReasoningItemContent::Text {
            text: "more raw reasoning".to_string(),
        }])),
        assistant_message("still nothing".to_string()),
    ];
    for item in &next_turn {
        append_with_generation(
            &store,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Paginated,
            item,
        )
        .await;
    }
    let extended = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect("resume after a second turn");
    assert_eq!(extended.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(
        extended.diagnostics.suffix_items,
        u64::try_from(next_turn.len()).expect("suffix items")
    );
    // The checkpoint published above already covers the first turn, so this resume replays only
    // the records written after it.
    let next_durable = extended
        .items
        .iter()
        .skip(1)
        .map(|item| serde_json::to_value(item).expect("serialize durable item"))
        .collect::<Vec<_>>();
    let next_expected = next_turn
        .iter()
        .map(|item| {
            serde_json::to_value(durable_readback_form(item)).expect("serialize expected item")
        })
        .collect::<Vec<_>>();
    assert_eq!(next_durable, next_expected);
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
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("unchanged extended resume");
    assert_eq!(unchanged.diagnostics.suffix_items, 0);
    assert_eq!(unchanged.items.len(), 1);
}

#[tokio::test]
async fn append_ancestry_verifies_the_checkpoint_owned_journal_anchor() {
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
    publish_loaded_state(&store, thread_id, &after_first_append).await;

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
async fn unsampled_rewrite_between_canonical_appends_is_loud() {
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

    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &user_message("first canonical suffix".to_string()),
    )
    .await;
    rewrite_unsampled_payload_byte(path.as_path());

    let error = crate::local::append_generation::begin_append(
        &store,
        thread_id,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &[user_message("rejected append".to_string())],
    )
    .expect_err("out-of-band rewrite before the next canonical append must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("outside the canonical append-generation contract"),
        "{error}"
    );
}

#[tokio::test]
async fn canonical_append_reads_only_suffix_plus_bounded_fences() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_012);
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

    let prefix_bytes = std::fs::metadata(path.as_path())
        .expect("large source metadata")
        .len();
    let (begin_io, finish_io) = append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &user_message("bounded canonical suffix".to_string()),
    )
    .await;
    let appended_bytes = std::fs::metadata(path.as_path())
        .expect("extended source metadata")
        .len()
        .saturating_sub(prefix_bytes);
    let source_bytes = begin_io.source_bytes.saturating_add(finish_io.source_bytes);

    assert_eq!(finish_io.suffix_bytes, appended_bytes);
    assert!(
        source_bytes <= appended_bytes.saturating_add(12 * FENCE_SAMPLE_BYTES as u64),
        "append inspected {source_bytes} bytes for a {appended_bytes}-byte suffix"
    );
    assert!(
        prefix_bytes > source_bytes.saturating_mul(4),
        "large {prefix_bytes}-byte prefix must not be replayed by a {source_bytes}-byte append inspection"
    );
}

#[tokio::test]
async fn forged_generation_or_foreign_checkpoint_ancestry_anchor_is_loud() {
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
    let valid_journal = std::fs::read(journal_path.as_path()).expect("read append generation");
    let mut journal: serde_json::Value =
        serde_json::from_slice(valid_journal.as_slice()).expect("decode append generation");
    let stable = journal
        .get_mut("stable")
        .and_then(serde_json::Value::as_object_mut)
        .expect("stable generation");
    stable.insert("generation".to_string(), serde_json::json!(1));
    stable.insert(
        "chain_sha256".to_string(),
        serde_json::json!("11".repeat(32)),
    );
    journal
        .as_object_mut()
        .expect("append generation journal")
        .insert("checkpoint_anchors".to_string(), serde_json::json!([]));
    std::fs::write(
        journal_path.as_path(),
        serde_json::to_vec(&journal).expect("encode forged append generation"),
    )
    .expect("write forged append generation");

    let error = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
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

    let mut journal: serde_json::Value =
        serde_json::from_slice(valid_journal.as_slice()).expect("decode valid generation");
    journal
        .get_mut("checkpoint_anchors")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|anchors| anchors.first_mut())
        .and_then(serde_json::Value::as_object_mut)
        .expect("checkpoint anchor")
        .insert(
            "checkpoint_thread_id".to_string(),
            serde_json::json!(ThreadId::new()),
        );
    std::fs::write(
        journal_path.as_path(),
        serde_json::to_vec(&journal).expect("encode foreign checkpoint owner"),
    )
    .expect("write foreign checkpoint owner");
    let error = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect_err("another checkpoint owner's anchor must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
    assert!(error.to_string().contains("ancestry anchor"), "{error}");
}

#[tokio::test]
async fn torn_canonical_append_rolls_back_and_allows_the_next_operation() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_013);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-13",
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
    let stable_bytes = std::fs::read(path.as_path()).expect("read stable source");

    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Legacy,
            &[user_message("torn append".to_string())],
        )
        .expect("begin torn append")
        .started
    );
    let mut file = OpenOptions::new()
        .append(true)
        .open(path.as_path())
        .expect("open torn source");
    file.write_all(br#"{"timestamp":"torn"#)
        .expect("write torn suffix");
    file.sync_all().expect("sync torn suffix");
    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("the current torn append must fail after rollback");
    assert!(
        matches!(error, ThreadStoreError::CanonicalAppendRolledBack { .. }),
        "{error}"
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read recovered source"),
        stable_bytes
    );

    let expected = user_message("expected canonical suffix".to_string());
    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Legacy,
            std::slice::from_ref(&expected),
        )
        .expect("begin fingerprinted append")
        .started
    );
    codex_rollout::append_rollout_item_to_path(
        path.as_path(),
        &user_message("unknown valid suffix".to_string()),
    )
    .await
    .expect("append unknown valid item");
    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("a valid but unknown suffix must roll back");
    assert!(
        matches!(
            &error,
            ThreadStoreError::CanonicalAppendRolledBack { reason }
                if reason.contains("canonical write intent")
        ),
        "{error}"
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read source after unknown suffix"),
        stable_bytes
    );

    let appended = user_message("retry after torn append".to_string());
    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &appended,
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
    .expect("resume after deterministic torn-append recovery");
    assert_eq!(resumed.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(resumed.diagnostics.suffix_items, 1);
}

/// A rejected canonical append truncates the durable suffix, but the live writer keeps the
/// position it already advanced to. Without a barrier every later append is silently rolled back
/// too, and the thread keeps running while nothing after the last durable record survives resume.
/// A record that differs only in fields the decoder drops is still a different durable write.
///
/// `cell_id` is `skip_deserializing`, so both payloads decode identically. A write intent that
/// normalized itself by decoding would accept the wrong record here; fingerprinting the durable
/// payload rejects it, and the correctly restored prefix survives.
#[tokio::test]
async fn a_suffix_differing_only_in_undecodable_fields_is_still_rejected() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_021);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-21",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let loaded = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load source");
    publish_loaded_state(&store, thread_id, &loaded).await;
    let stable_bytes = std::fs::read(path.as_path()).expect("read stable source");

    let intended = message_with_host_metadata("intended-cell");
    let written = message_with_host_metadata("substituted-cell");
    assert_eq!(
        serde_json::to_value(durable_readback_form(&intended)).expect("decode intended"),
        serde_json::to_value(durable_readback_form(&written)).expect("decode written"),
        "the two payloads must be indistinguishable after decoding"
    );

    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Paginated,
            std::slice::from_ref(&intended),
        )
        .expect("begin append")
        .started
    );
    codex_rollout::append_rollout_item_to_path(path.as_path(), &written)
        .await
        .expect("write the substituted record");
    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("a substituted durable payload must be rejected");
    assert!(
        matches!(
            &error,
            ThreadStoreError::CanonicalAppendRolledBack { reason }
                if reason.contains("canonical write intent")
        ),
        "{error}"
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read source after rollback"),
        stable_bytes
    );
}

/// An append whose durable write never completed must not be followed by a stale write.
///
/// The recorder still holds whatever it buffered, so a later flush would place those items after a
/// durable position this writer can no longer justify. Nothing may write again, a checkpoint may
/// not be published, and shutdown must still release the handle without writing.
#[cfg(unix)]
#[tokio::test]
async fn an_unresolved_durable_write_stops_the_writer_without_losing_cleanup() {
    use std::os::unix::fs::PermissionsExt;

    let home = TempDir::new().expect("temp dir");
    let thread_id = ThreadId::new();
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    store
        .create_thread(CreateThreadParams {
            session_id: thread_id.into(),
            thread_id,
            extra_config: None,
            forked_from_id: None,
            parent_thread_id: None,
            source: SessionSource::Exec,
            thread_source: None,
            originator: "test_originator".to_string(),
            base_instructions: codex_protocol::models::BaseInstructions::default(),
            dynamic_tools: Vec::new(),
            selected_capability_roots: Vec::new(),
            multi_agent_version: None,
            history_mode: ThreadHistoryMode::Legacy,
            history_base: None,
            subagent_history_start_ordinal: None,
            initial_window_id: "window-1".to_string(),
            metadata: ThreadPersistenceMetadata {
                cwd: Some(home.path().to_path_buf()),
                model_provider: "test-provider".to_string(),
                memory_mode: ThreadMemoryMode::Enabled,
            },
        })
        .await
        .expect("create thread");

    // The rollout materializes lazily, so making its directory unwritable is a real IO failure at
    // the writer rather than an injected hook.
    let sessions = home.path().join("sessions");
    std::fs::create_dir_all(sessions.as_path()).expect("sessions dir");
    let original = std::fs::metadata(sessions.as_path())
        .expect("sessions metadata")
        .permissions();
    std::fs::set_permissions(sessions.as_path(), std::fs::Permissions::from_mode(0o555))
        .expect("make sessions read-only");

    let first = store
        .append_items(AppendThreadItemsParams {
            thread_id,
            items: vec![user_message("never reached disk".to_string())],
        })
        .await
        .expect_err("the durable write cannot complete");

    std::fs::set_permissions(sessions.as_path(), original).expect("restore sessions permissions");

    // Even with the filesystem healthy again, this writer must not place its buffered items.
    let second = store
        .append_items(AppendThreadItemsParams {
            thread_id,
            items: vec![user_message("must not be written either".to_string())],
        })
        .await
        .expect_err("a stopped writer must not resume writing");
    assert!(
        matches!(second, ThreadStoreError::CanonicalAppendRolledBack { .. }),
        "first={first}, second={second}"
    );
    let published = store
        .publish_materialized_resume_state(PublishMaterializedResumeParams {
            thread_id,
            fence: MaterializedResumePublicationFence::Current {
                rollout_path: home.path().join("sessions/unresolved.jsonl"),
                history_mode: ThreadHistoryMode::Legacy,
            },
            state: state(),
            max_state_bytes: 64 * 1024,
        })
        .await
        .expect_err("a checkpoint must not run ahead of an unresolved rollout");
    assert_eq!(published.to_string(), second.to_string());

    // Cleanup still happens, and it writes nothing.
    let before = std::fs::read_dir(sessions.as_path())
        .expect("read sessions")
        .count();
    store
        .shutdown_thread(thread_id)
        .await
        .expect("a stopped writer must still release its handle");
    assert_eq!(
        std::fs::read_dir(sessions.as_path())
            .expect("read sessions after shutdown")
            .count(),
        before
    );
    assert!(matches!(
        store
            .append_items(AppendThreadItemsParams {
                thread_id,
                items: vec![user_message("after teardown".to_string())],
            })
            .await,
        Err(ThreadStoreError::ThreadNotFound { .. })
    ));
}

/// A stable generation predates this release's pending fingerprint and must keep working.
///
/// The discriminator lives on the pending record only, so a journal that is not mid-append is
/// byte-identical to what an earlier release wrote. It must accept new writes and resume with its
/// identity, chain and anchors intact rather than being treated as defective history.
#[tokio::test]
async fn a_stable_generation_without_a_pending_fingerprint_keeps_working() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_023);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-23",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let loaded = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load source");
    publish_loaded_state(&store, thread_id, &loaded).await;

    // A settled journal carries no fingerprint at all, so an earlier release wrote exactly this.
    let journal_path = crate::local::append_generation::journal_path(&store, thread_id);
    let settled = std::fs::read_to_string(journal_path.as_path()).expect("read journal");
    assert!(!settled.contains("fingerprint"), "{settled}");
    let identity: serde_json::Value = serde_json::from_str(settled.as_str()).expect("journal json");

    append_with_generation(
        &store,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Paginated,
        &user_message("written against the settled generation".to_string()),
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
    .expect("resume after writing against a settled generation");
    assert_eq!(resumed.diagnostics.outcome, ResumeCheckpointOutcome::Hit);
    assert_eq!(resumed.diagnostics.suffix_items, 1);

    let advanced: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(journal_path.as_path())
            .expect("read advanced journal")
            .as_str(),
    )
    .expect("advanced journal json");
    assert_eq!(advanced["generation_id"], identity["generation_id"]);
    assert_eq!(advanced["rollout_id"], identity["rollout_id"]);
    assert_eq!(
        advanced["canonical_rollout_path"],
        identity["canonical_rollout_path"]
    );
    // The anchor keeps its own ancestry while its descendant pointer follows the new write.
    for field in ["anchor_id", "checkpoint_thread_id", "generation", "chain_sha256"] {
        assert_eq!(
            advanced["checkpoint_anchors"][0][field],
            identity["checkpoint_anchors"][0][field],
            "{field}"
        );
    }
    assert_ne!(
        advanced["checkpoint_anchors"][0]["descendant_generation"],
        identity["checkpoint_anchors"][0]["descendant_generation"]
    );
    assert_ne!(advanced["stable"], identity["stable"]);
}

/// A pending append recorded by a superseded fingerprint definition must not be verified here.
#[tokio::test]
async fn a_superseded_pending_fingerprint_is_rejected_rather_than_reinterpreted() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_022);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-22",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let loaded = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load source");
    publish_loaded_state(&store, thread_id, &loaded).await;
    let stable_bytes = std::fs::read(path.as_path()).expect("read stable source");

    let appended = user_message("written by the previous release".to_string());
    assert!(
        crate::local::append_generation::begin_append(
            &store,
            thread_id,
            thread_id,
            path.as_path(),
            ThreadHistoryMode::Paginated,
            std::slice::from_ref(&appended),
        )
        .expect("begin append")
        .started
    );
    // Rewrite the journal as an earlier release left it: a pending fingerprint with no definition.
    let journal_path = crate::local::append_generation::journal_path(&store, thread_id);
    let mut journal: serde_json::Value = serde_json::from_slice(
        std::fs::read(journal_path.as_path())
            .expect("read journal")
            .as_slice(),
    )
    .expect("journal is json");
    let pending = journal
        .get_mut("pending")
        .and_then(|pending| pending.get_mut("evidence"))
        .and_then(serde_json::Value::as_object_mut)
        .expect("pending evidence");
    assert!(pending.remove("fingerprint").is_some());
    std::fs::write(
        journal_path.as_path(),
        serde_json::to_vec(&journal).expect("encode journal"),
    )
    .expect("write journal");

    let journal_bytes = std::fs::read(
        crate::local::append_generation::journal_path(&store, thread_id).as_path(),
    )
    .expect("read superseded journal");

    // Even with nothing written yet the refusal must not clear the pending evidence.
    let error = crate::local::append_generation::load_current(&store, thread_id, path.as_path())
        .expect_err("an empty suffix must not silently clear a superseded pending");
    assert!(error.to_string().contains("superseded release"), "{error}");
    assert_eq!(
        std::fs::read(
            crate::local::append_generation::journal_path(&store, thread_id).as_path()
        )
        .expect("read journal after empty-suffix refusal"),
        journal_bytes
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read source after empty-suffix refusal"),
        stable_bytes
    );

    codex_rollout::append_rollout_item_to_path(path.as_path(), &appended)
        .await
        .expect("append the pending suffix");
    let suffix_bytes = std::fs::read(path.as_path()).expect("read source with pending suffix");
    assert_ne!(suffix_bytes, stable_bytes);

    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("a superseded fingerprint cannot be verified");
    assert!(
        error.to_string().contains("superseded release"),
        "{error}"
    );
    // Refusal must not be destructive: the unverifiable suffix is left exactly where it is, so
    // nothing is discarded on rules that never applied to it.
    assert_eq!(
        std::fs::read(path.as_path()).expect("read source after refusal"),
        suffix_bytes
    );
    // The refusal is loud on every path, including plain recovery, not only when finishing.
    let error = crate::local::append_generation::load_current(
        &store,
        thread_id,
        path.as_path(),
    )
    .expect_err("recovery must refuse a superseded pending too");
    assert!(
        error.to_string().contains("superseded release"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read source after recovery refusal"),
        suffix_bytes
    );
    // Nothing about the journal moved either: no stable advance, no anchor edit, no evidence clear.
    assert_eq!(
        std::fs::read(
            crate::local::append_generation::journal_path(&store, thread_id).as_path()
        )
        .expect("read journal after refusals"),
        journal_bytes
    );
}

#[tokio::test]
async fn a_rolled_back_append_stops_the_live_writer_instead_of_poisoning_it() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_020);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-20",
        uuid,
        ThreadHistoryMode::Paginated,
    )
    .expect("write session file");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    store
        .resume_thread(ResumeThreadParams {
            thread_id,
            rollout_path: Some(path.clone()),
            history: None,
            include_archived: false,
            metadata: ThreadPersistenceMetadata {
                cwd: Some(home.path().to_path_buf()),
                model_provider: "test-provider".to_string(),
                memory_mode: ThreadMemoryMode::Enabled,
            },
        })
        .await
        .expect("open the live writer");

    // The live writer captured its ordinal when it opened the rollout. Another writer extending
    // the same file leaves it one record behind, which is the state a rollback also produces.
    codex_rollout::append_rollout_item_to_path(
        path.as_path(),
        &user_message("out-of-band record".to_string()),
    )
    .await
    .expect("extend the rollout out of band");
    let loaded = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load the extended source");
    publish_loaded_state(&store, thread_id, &loaded).await;
    let durable_bytes = std::fs::read(path.as_path()).expect("read durable transcript");

    let rejected = store
        .append_items(AppendThreadItemsParams {
            thread_id,
            items: vec![user_message("first lost item".to_string())],
        })
        .await
        .expect_err("a suffix written at a stale ordinal must be rejected");
    assert!(
        matches!(
            &rejected,
            ThreadStoreError::CanonicalAppendRolledBack { reason }
                if reason.contains("pending append ordinal")
        ),
        "{rejected}"
    );
    assert_eq!(
        std::fs::read(path.as_path()).expect("read transcript after rollback"),
        durable_bytes
    );

    let repeated = store
        .append_items(AppendThreadItemsParams {
            thread_id,
            items: vec![user_message("second lost item".to_string())],
        })
        .await
        .expect_err("a writer that lost durable history must not keep appending");
    assert_eq!(repeated.to_string(), rejected.to_string());
    assert_eq!(
        std::fs::read(path.as_path()).expect("read transcript after the barrier"),
        durable_bytes
    );

    let source = loaded
        .materialized_resume
        .as_ref()
        .expect("materialization fence")
        .source
        .clone();
    let published = store
        .publish_materialized_resume_state(PublishMaterializedResumeParams {
            thread_id,
            fence: MaterializedResumePublicationFence::Loaded(Box::new(source)),
            state: state(),
            max_state_bytes: 64 * 1024,
        })
        .await
        .expect_err("a checkpoint must not present lost history as durable");
    assert_eq!(published.to_string(), rejected.to_string());
}

#[tokio::test]
async fn parent_checkpoint_publication_preserves_a_fork_owned_ancestry_anchor() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_014);
    let parent_thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let child_thread_id =
        ThreadId::from_string(&Uuid::from_u128(/*v*/ 4_015).to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-14",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write parent source");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let parent = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id: parent_thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("load parent source");
    publish_loaded_state(&store, parent_thread_id, &parent).await;
    let child_state = state();
    store
        .publish_materialized_resume_state(PublishMaterializedResumeParams {
            thread_id: child_thread_id,
            fence: MaterializedResumePublicationFence::Current {
                rollout_path: path.clone(),
                history_mode: ThreadHistoryMode::Legacy,
            },
            state: child_state.clone(),
            max_state_bytes: 64 * 1024 * 1024,
        })
        .await
        .expect("publish fork-owned source anchor");

    let unchanged_parent = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id: parent_thread_id,
            include_archived: false,
            rollout_path: Some(path.clone()),
        },
    )
    .await
    .expect("reload parent");
    publish_loaded_state(&store, parent_thread_id, &unchanged_parent).await;
    append_with_generation(
        &store,
        parent_thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
        &user_message("parent suffix after fork".to_string()),
    )
    .await;

    let child_source = capture_current_source(
        &store,
        child_thread_id,
        path.as_path(),
        ThreadHistoryMode::Legacy,
    )
    .await
    .expect("capture fork source");
    let child = load_checkpoint(&store, child_source.source)
        .await
        .expect("validate fork anchor")
        .expect("fork checkpoint");
    assert_eq!(child.materialized_resume.state, Some(child_state));
    assert_eq!(child.suffix_segments.len(), 1);
    assert!(child.suffix_segments[0].end_byte_offset > child.suffix_segments[0].start_byte_offset);
}

#[tokio::test]
async fn obsolete_resume_namespaces_are_an_explicit_first_use_miss() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 4_016);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_session_file_with_history_mode(
        home.path(),
        "2025-01-03T15-00-16",
        uuid,
        ThreadHistoryMode::Legacy,
    )
    .expect("write source");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    for (directory, contents) in [
        ("materialized_resume_state_v1", b"obsolete-state".as_slice()),
        (
            "materialized_resume_state_v4",
            b"superseded-state".as_slice(),
        ),
        (
            "rollout_append_generation_v1",
            b"obsolete-generation".as_slice(),
        ),
    ] {
        let artifact = home
            .path()
            .join(directory)
            .join(format!("{thread_id}.json"));
        std::fs::create_dir_all(artifact.parent().expect("obsolete directory"))
            .expect("create obsolete namespace");
        std::fs::write(artifact, contents).expect("write obsolete derived state");
    }

    let loaded = load_latest_model_context(
        &store,
        LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(path),
        },
    )
    .await
    .expect("new namespace must bootstrap independently");
    assert_eq!(loaded.diagnostics.outcome, ResumeCheckpointOutcome::Miss);
    publish_loaded_state(&store, thread_id, &loaded).await;
    assert!(checkpoint_path(&store, thread_id).exists());
    assert!(crate::local::append_generation::journal_path(&store, thread_id).exists());
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

    let append = crate::local::append_generation::begin_append(
        &store,
        thread_id,
        thread_id,
        path.as_path(),
        ThreadHistoryMode::Paginated,
        &[user_message("expected ordinal".to_string())],
    )
    .expect("begin invalid canonical append");
    assert!(append.started);
    let transcript_before = std::fs::read(path.as_path()).expect("read stable transcript");
    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open paginated rollout");
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

    let error = crate::local::append_generation::finish_append(&store, thread_id)
        .expect_err("ordinal gap must fail and roll back");
    assert!(error.to_string().contains("rolled back"), "{error}");
    assert!(error.to_string().contains("ordinal"), "{error}");
    assert_eq!(
        std::fs::read(path.as_path()).expect("read restored transcript"),
        transcript_before
    );
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
