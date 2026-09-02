use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;

use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
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
        estimated_prefill_input_tokens: None,
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
            source,
            state: state(),
            max_state_bytes: 64 * 1024 * 1024,
        })
        .await
        .expect("publish materialized state");
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
    codex_rollout::append_rollout_item_to_path(&path, &appended)
        .await
        .expect("append suffix");
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

    publish_loaded_state(&store, thread_id, &first).await;
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

    publish_loaded_state(&store, thread_id, &first).await;
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

    publish_loaded_state(&store, thread_id, &first).await;
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
    std::fs::create_dir_all(unrelated_artifact.parent().expect("artifact directory"))
        .expect("create artifact directory");
    std::fs::write(&unrelated_artifact, b"unrelated checkpoint")
        .expect("write unrelated checkpoint");
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
        assert!(artifact.exists());
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
        assert_eq!(
            std::fs::read(&unrelated_artifact).expect("read unrelated checkpoint"),
            b"unrelated checkpoint"
        );
    }
}
