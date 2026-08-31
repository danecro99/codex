use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::items::TurnItem;
use codex_protocol::items::UserMessageItem;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::RateLimitWindow;
use codex_protocol::protocol::SandboxPolicy;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::protocol::TokenCountEvent;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::protocol::UserMessageEvent;
use codex_protocol::user_input::UserInput;
use codex_rollout::CompactedItem;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutLine;
use pretty_assertions::assert_eq;
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::ThreadStore;
use crate::local::test_support::test_config;
use crate::local::test_support::write_session_file_with_history_mode;

#[tokio::test]
async fn loads_latest_checkpoint_with_required_turn_metadata() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1001);
    let thread_id = codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id");
    write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-00",
        uuid,
        [
            turn_started("turn-1"),
            user_message("older turn"),
            completed_user_message("turn-1", "older turn"),
            turn_context(home.path(), "turn-1"),
            compacted("older checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            turn_started("turn-2"),
            user_message("latest turn"),
            completed_user_message("turn-2", "latest turn"),
            turn_context(home.path(), "turn-2"),
            compacted("latest checkpoint", Some(Vec::new())),
            turn_complete("turn-2"),
        ],
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("load model context");

    assert!(matches!(
        context.items.first(),
        Some(RolloutItem::SessionMeta(_))
    ));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::Compacted(compacted) if compacted.message == "latest checkpoint")
    }));
    assert!(!context.items.iter().any(|item| {
        matches!(item, RolloutItem::Compacted(compacted) if compacted.message == "older checkpoint")
    }));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-2"))
    }));
}

#[tokio::test]
async fn loads_bounded_legacy_context_from_exact_rollout_path() {
    let home = TempDir::new().expect("temp dir");
    let source_path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-06",
        Uuid::from_u128(/*v*/ 1013),
        [user_message("copied source")],
    );
    let copied_session_meta = codex_rollout::read_session_meta_line(source_path.as_path())
        .await
        .expect("read copied source metadata");
    let uuid = Uuid::from_u128(/*v*/ 1008);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let selected_path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-07",
        uuid,
        [
            turn_started("turn-1"),
            user_message("older turn"),
            legacy_user_message_event("older turn"),
            turn_context(home.path(), "turn-1"),
            compacted("older checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            turn_started("turn-2"),
            user_message("latest turn"),
            legacy_user_message_event("latest turn"),
            turn_context(home.path(), "turn-2"),
            RolloutItem::SessionMeta(copied_session_meta),
            compacted("selected checkpoint", Some(Vec::new())),
            turn_complete("turn-2"),
        ],
    );
    write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-08",
        uuid,
        [
            turn_started("other-turn"),
            user_message("other rollout"),
            legacy_user_message_event("other rollout"),
            turn_context(home.path(), "other-turn"),
            compacted("other checkpoint", Some(Vec::new())),
            turn_complete("other-turn"),
        ],
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: Some(selected_path),
        })
        .await
        .expect("load selected legacy model context");

    assert!(matches!(
        context.items.first(),
        Some(RolloutItem::SessionMeta(meta)) if meta.meta.id == thread_id
    ));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::Compacted(compacted) if compacted.message == "selected checkpoint")
    }));
    assert!(!context.items.iter().any(|item| {
        matches!(item, RolloutItem::Compacted(compacted) if compacted.message == "older checkpoint" || compacted.message == "other checkpoint")
    }));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-2"))
    }));
}

#[tokio::test]
async fn rejects_rollout_path_for_a_different_thread() {
    let home = TempDir::new().expect("temp dir");
    let rollout_uuid = Uuid::from_u128(/*v*/ 1011);
    let rollout_path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-11",
        rollout_uuid,
        [user_message("thread one")],
    );
    let expected_thread_id = ThreadId::from_string(&Uuid::from_u128(/*v*/ 1012).to_string())
        .expect("expected thread id");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let error = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: expected_thread_id,
            include_archived: false,
            rollout_path: Some(rollout_path),
        })
        .await
        .expect_err("mismatched rollout identity should fail");

    assert!(error.to_string().contains("belongs to thread"), "{error}");
}

#[tokio::test]
async fn fork_context_excludes_items_after_frozen_cutoff() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1007);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-06",
        uuid,
        [turn_started("frozen-turn"), user_message("frozen message")],
    );
    let history_base =
        history_position(path.as_path(), thread_id, /*end_ordinal_exclusive*/ 3);
    append_items(path.as_path(), [user_message("later message")]);
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let lineage = store
        .resolve_rollout_lineage(thread_id)
        .await
        .expect("resolve source lineage");
    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .expect("read source metadata");

    let context = load_for_fork(lineage, Some(history_base))
        .await
        .expect("load frozen fork context");

    let expected = vec![
        RolloutItem::SessionMeta(session_meta),
        turn_started("frozen-turn"),
        user_message("frozen message"),
    ];
    assert_eq!(
        serde_json::to_value(context).expect("serialize fork context"),
        serde_json::to_value(expected).expect("serialize expected fork context")
    );
}

#[tokio::test]
async fn loads_turn_metadata_across_an_older_checkpoint() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1006);
    let thread_id = codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id");
    write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-05",
        uuid,
        [
            turn_started("turn-0"),
            user_message("oldest turn"),
            completed_user_message("turn-0", "oldest turn"),
            turn_context(home.path(), "turn-0"),
            turn_complete("turn-0"),
            turn_started("turn-1"),
            user_message("metadata turn"),
            completed_user_message("turn-1", "metadata turn"),
            turn_context(home.path(), "turn-1"),
            compacted("older checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            turn_started("turn-2"),
            compacted("latest checkpoint", Some(Vec::new())),
            turn_complete("turn-2"),
        ],
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("load model context");

    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::Compacted(compacted) if compacted.message == "latest checkpoint")
    }));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-1"))
    }));
    assert!(!context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-0"))
    }));
}

#[tokio::test]
async fn rejects_unsupported_compaction_without_a_safe_cutoff() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1002);
    let path = write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-01",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            completed_user_message("turn-1", "turn"),
            turn_context(home.path(), "turn-1"),
            compacted("usable checkpoint", Some(Vec::new())),
            compacted("legacy checkpoint", /*replacement_history*/ None),
            turn_complete("turn-1"),
        ],
    );

    assert_model_context_scan_fails(
        home.path(),
        path.as_path(),
        "does not contain a safe bounded model-context checkpoint",
    )
    .await;
}

#[tokio::test]
async fn rejects_legacy_history_without_a_valid_cutoff() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1009);
    let path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-10",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            legacy_user_message_event("turn"),
            turn_context(home.path(), "turn-1"),
            turn_complete("turn-1"),
        ],
    );

    assert_model_context_scan_fails(
        home.path(),
        path.as_path(),
        "does not contain a safe bounded model-context checkpoint",
    )
    .await;
}

#[tokio::test]
async fn loads_bounded_paginated_history_from_its_durable_origin() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1003);
    let path = write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-02",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            completed_user_message("turn-1", "turn"),
            turn_context(home.path(), "turn-1"),
            turn_complete("turn-1"),
        ],
    );

    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id"),
            include_archived: false,
            rollout_path: Some(path),
        })
        .await
        .expect("bounded paginated origin should load");

    assert!(matches!(
        context.items.first(),
        Some(RolloutItem::SessionMeta(_))
    ));
    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::ResponseItem(item) if matches!(
            &item.item,
            ResponseItem::Message { role, .. } if role == "user"
        ))
    }));
}

#[tokio::test]
async fn resumes_fresh_written_rollout_with_floating_point_rate_limits() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1015);
    let items = [
        turn_started("turn-1"),
        user_message("fresh turn"),
        completed_user_message("turn-1", "fresh turn"),
        turn_context(home.path(), "turn-1"),
        floating_point_token_count(),
        turn_complete("turn-1"),
    ];
    let path = write_paginated_rollout(home.path(), "2025-01-03T13-00-13", uuid, items.clone());
    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: session_meta.meta.id,
            include_archived: false,
            rollout_path: Some(path),
        })
        .await
        .expect("fresh-written rollout should resume");

    let expected = std::iter::once(RolloutItem::SessionMeta(session_meta))
        .chain(items)
        .collect::<Vec<_>>();
    assert_eq!(
        serde_json::to_value(context.items).expect("serialize resumed context"),
        serde_json::to_value(expected).expect("serialize expected context")
    );
}

#[tokio::test]
async fn rejects_model_context_over_the_token_limit() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1014);
    let path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-12",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            legacy_user_message_event("turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
        ],
    );
    let bounded_message = "x".repeat((codex_rollout::MODEL_CONTEXT_MAX_ITEM_TOKENS - 1_000) * 4);
    append_repeated_item(
        path.as_path(),
        user_message(&bounded_message),
        codex_rollout::MODEL_CONTEXT_MAX_TOKENS
            / (codex_rollout::MODEL_CONTEXT_MAX_ITEM_TOKENS - 1_000)
            + 2,
    );

    assert_model_context_scan_fails(home.path(), path.as_path(), "exceeds the token limit").await;
}

#[tokio::test]
async fn rejects_single_model_context_item_over_the_token_limit() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1017);
    let oversized_message = "x".repeat(codex_rollout::MODEL_CONTEXT_MAX_ITEM_TOKENS * 4 + 1);
    let path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-15",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            legacy_user_message_event("turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            user_message(&oversized_message),
        ],
    );

    assert_model_context_scan_fails(home.path(), path.as_path(), "exceeds the item token limit")
        .await;
}

#[tokio::test]
async fn loads_compacted_context_after_scanning_replaced_oversized_item() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1018);
    let oversized_message = "x".repeat(codex_rollout::MODEL_CONTEXT_MAX_ITEM_TOKENS * 4 + 1);
    let items = [
        turn_started("turn-1"),
        user_message("turn"),
        legacy_user_message_event("turn"),
        turn_context(home.path(), "turn-1"),
        user_message(&oversized_message),
        turn_complete("turn-1"),
        compacted("latest checkpoint", Some(Vec::new())),
    ];
    let path = write_legacy_rollout(home.path(), "2025-01-03T13-00-16", uuid, items.clone());
    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: session_meta.meta.id,
            include_archived: false,
            rollout_path: Some(path),
        })
        .await
        .expect("compacted context should admit superseded oversized items during its scan");

    let expected = std::iter::once(RolloutItem::SessionMeta(session_meta))
        .chain(items)
        .collect::<Vec<_>>();
    assert_eq!(
        serde_json::to_value(context.items).expect("serialize resumed context"),
        serde_json::to_value(expected).expect("serialize expected context")
    );
}

#[tokio::test]
async fn rejects_model_context_over_the_item_limit() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1016);
    let path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-14",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            legacy_user_message_event("turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
        ],
    );
    append_repeated_item(
        path.as_path(),
        turn_complete("later-turn"),
        codex_rollout::MODEL_CONTEXT_MAX_ITEMS,
    );

    assert_model_context_scan_fails(home.path(), path.as_path(), "exceeds the item limit").await;
}

#[tokio::test]
async fn rejects_compressed_model_context_without_a_bounded_stream() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1015);
    let path = write_legacy_rollout(
        home.path(),
        "2025-01-03T13-00-13",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            legacy_user_message_event("turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
        ],
    );
    let compressed_path = path.with_extension("jsonl.zst");
    std::fs::write(&compressed_path, b"not a zstd stream")
        .expect("write intentionally unreadable compressed rollout");
    std::fs::remove_file(&path).expect("remove plain rollout");

    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let error = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id"),
            include_archived: false,
            rollout_path: Some(compressed_path),
        })
        .await
        .expect_err("compressed resume should require a bounded stream");

    assert!(
        matches!(
            error,
            ThreadStoreError::Unsupported {
                operation: "bounded_model_context_from_compressed_rollout"
            }
        ),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn uses_agent_message_turn_context_without_scanning_older_turn() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1004);
    let thread_id = codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id");
    write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-03",
        uuid,
        [
            turn_started("turn-1"),
            user_message("older turn"),
            completed_user_message("turn-1", "older turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            turn_started("turn-2"),
            turn_context(home.path(), "turn-2"),
            agent_message("child done"),
            turn_complete("turn-2"),
        ],
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("load model context");

    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-2"))
    }));
    assert!(!context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-1"))
    }));
}

#[tokio::test]
async fn ignores_contextual_user_messages_when_selecting_turn_context() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1005);
    let thread_id = codex_protocol::ThreadId::from_string(&uuid.to_string()).expect("thread id");
    write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-04",
        uuid,
        [
            turn_started("turn-1"),
            user_message("real user turn"),
            completed_user_message("turn-1", "real user turn"),
            turn_context(home.path(), "turn-1"),
            compacted("checkpoint", Some(Vec::new())),
            turn_complete("turn-1"),
            turn_started("turn-2"),
            contextual_user_message(),
            turn_context(home.path(), "turn-2"),
            turn_complete("turn-2"),
        ],
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("load model context");

    assert!(context.items.iter().any(|item| {
        matches!(item, RolloutItem::TurnContext(context) if context.turn_id.as_deref() == Some("turn-1"))
    }));
}

#[tokio::test]
async fn replays_nested_archived_lineage_from_frozen_prefix() {
    let home = TempDir::new().expect("temp dir");
    let root_uuid = Uuid::from_u128(/*v*/ 2001);
    let root_id = ThreadId::from_string(&root_uuid.to_string()).expect("root id");
    let root_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-00",
        root_uuid,
        [
            user_message("root before checkpoint"),
            compacted("root checkpoint", Some(Vec::new())),
            turn_started("root-excluded"),
            user_message("root after cutoff"),
        ],
    );
    let archived_root = home
        .path()
        .join("archived_sessions")
        .join(root_path.file_name().expect("root filename"));
    std::fs::create_dir_all(archived_root.parent().expect("archive parent"))
        .expect("create archive directory");
    std::fs::rename(root_path, &archived_root).expect("archive root rollout");

    let middle_uuid = Uuid::from_u128(/*v*/ 2002);
    let middle_id = ThreadId::from_string(&middle_uuid.to_string()).expect("middle id");
    let middle_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-01",
        middle_uuid,
        [
            turn_started("middle-turn"),
            user_message("middle inherited"),
            completed_user_message("middle-turn", "middle inherited"),
            turn_context(home.path(), "middle-turn"),
            turn_complete("middle-turn"),
        ],
    );
    set_history_base(
        middle_path.as_path(),
        history_position(
            archived_root.as_path(),
            root_id,
            /*end_ordinal_exclusive*/ 3,
        ),
    );

    let child_uuid = Uuid::from_u128(/*v*/ 2003);
    let child_id = ThreadId::from_string(&child_uuid.to_string()).expect("child id");
    let child_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-02",
        child_uuid,
        [
            turn_started("child-turn"),
            user_message("child local"),
            completed_user_message("child-turn", "child local"),
            turn_context(home.path(), "child-turn"),
            turn_complete("child-turn"),
        ],
    );
    set_history_base(
        child_path.as_path(),
        history_position(
            middle_path.as_path(),
            middle_id,
            /*end_ordinal_exclusive*/ 6,
        ),
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: child_id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect("load lineage model context");

    assert!(matches!(
        context.items.first(),
        Some(RolloutItem::SessionMeta(meta)) if meta.meta.id == child_id
    ));
    let child_meta = codex_rollout::read_session_meta_line(child_path.as_path())
        .await
        .expect("read child metadata");
    let expected = vec![
        RolloutItem::SessionMeta(child_meta),
        compacted("root checkpoint", Some(Vec::new())),
        turn_started("middle-turn"),
        user_message("middle inherited"),
        completed_user_message("middle-turn", "middle inherited"),
        turn_context(home.path(), "middle-turn"),
        turn_complete("middle-turn"),
        turn_started("child-turn"),
        user_message("child local"),
        completed_user_message("child-turn", "child local"),
        turn_context(home.path(), "child-turn"),
        turn_complete("child-turn"),
    ];
    assert_eq!(
        serde_json::to_value(context.items).expect("serialize context"),
        serde_json::to_value(expected).expect("serialize expected context")
    );
}

fn write_paginated_rollout<const N: usize>(
    home: &Path,
    timestamp: &str,
    uuid: Uuid,
    items: [RolloutItem; N],
) -> PathBuf {
    let path =
        write_session_file_with_history_mode(home, timestamp, uuid, ThreadHistoryMode::Paginated)
            .expect("write session file");
    append_items(path.as_path(), items);
    path
}

fn write_legacy_rollout<const N: usize>(
    home: &Path,
    timestamp: &str,
    uuid: Uuid,
    items: [RolloutItem; N],
) -> PathBuf {
    let path =
        write_session_file_with_history_mode(home, timestamp, uuid, ThreadHistoryMode::Legacy)
            .expect("write session file");
    append_items(path.as_path(), items);
    path
}

fn write_ordinaled_paginated_rollout<const N: usize>(
    home: &Path,
    timestamp: &str,
    uuid: Uuid,
    items: [RolloutItem; N],
) -> PathBuf {
    let path =
        write_session_file_with_history_mode(home, timestamp, uuid, ThreadHistoryMode::Paginated)
            .expect("write session file");
    let mut file = OpenOptions::new()
        .append(true)
        .open(path.as_path())
        .expect("open session file");
    for (index, item) in items.into_iter().enumerate() {
        let line = RolloutLine {
            timestamp: "2025-01-03T13:00:01Z".to_string(),
            ordinal: Some(u64::try_from(index).expect("fixture index fits u64") + 1),
            item,
        };
        writeln!(
            file,
            "{}",
            serde_json::to_string(&line).expect("serialize line")
        )
        .expect("append rollout line");
    }
    path
}

fn set_history_base(path: &Path, history_base: HistoryPosition) {
    let contents = std::fs::read_to_string(path).expect("read rollout");
    let mut lines = contents.lines();
    let mut head: serde_json::Value =
        serde_json::from_str(lines.next().expect("session meta line")).expect("parse head");
    head["payload"]["history_base"] =
        serde_json::to_value(history_base).expect("serialize history base");
    let mut updated = serde_json::to_string(&head).expect("serialize head");
    for line in lines {
        updated.push('\n');
        updated.push_str(line);
    }
    updated.push('\n');
    std::fs::write(path, updated).expect("write history base");
}

fn history_position(
    path: &Path,
    thread_id: ThreadId,
    end_ordinal_exclusive: u64,
) -> HistoryPosition {
    HistoryPosition {
        thread_id,
        end_ordinal_exclusive,
        end_byte_offset: rollout_end_byte_offset(path, end_ordinal_exclusive),
    }
}

fn rollout_end_byte_offset(path: &Path, end_ordinal_exclusive: u64) -> u64 {
    let contents = std::fs::read(path).expect("read rollout");
    let mut byte_offset = 0_u64;
    for line in contents.split_inclusive(|byte| *byte == b'\n') {
        let parsed: RolloutLine =
            serde_json::from_slice(line).expect("parse rollout line for byte offset");
        if parsed.ordinal == Some(end_ordinal_exclusive) {
            return byte_offset;
        }
        byte_offset += u64::try_from(line.len()).expect("line length fits u64");
    }
    byte_offset
}

async fn assert_model_context_scan_fails(home: &Path, path: &Path, expected_message: &str) {
    let session_meta = codex_rollout::read_session_meta_line(path)
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home), /*state_db*/ None);
    let error = store
        .load_latest_model_context(LoadModelContextParams {
            thread_id: session_meta.meta.id,
            include_archived: false,
            rollout_path: None,
        })
        .await
        .expect_err("model context scan should fail");

    assert!(
        error.to_string().contains(expected_message),
        "unexpected error: {error}"
    );
}

fn append_items<const N: usize>(path: &Path, items: [RolloutItem; N]) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open session file");
    for item in items {
        let line = RolloutLine {
            timestamp: "2025-01-03T13:00:01Z".to_string(),
            ordinal: None,
            item,
        };
        writeln!(
            file,
            "{}",
            serde_json::to_string(&line).expect("serialize line")
        )
        .expect("append rollout line");
    }
}

fn append_repeated_item(path: &Path, item: RolloutItem, count: usize) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open session file");
    let line = RolloutLine {
        timestamp: "2025-01-03T13:00:01Z".to_string(),
        ordinal: None,
        item,
    };
    let serialized = serde_json::to_string(&line).expect("serialize line");
    for _ in 0..count {
        writeln!(file, "{serialized}").expect("append rollout line");
    }
}

fn turn_started(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
        turn_id: turn_id.to_string(),
        trace_id: None,
        started_at: None,
        model_context_window: Some(128_000),
        collaboration_mode_kind: Default::default(),
    }))
}

fn turn_complete(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
        turn_id: turn_id.to_string(),
        last_agent_message: None,
        error: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        time_to_first_token_ms: None,
    }))
}

fn floating_point_token_count() -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TokenCount(TokenCountEvent {
        info: None,
        rate_limits: Some(RateLimitSnapshot {
            limit_id: None,
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 42.0,
                window_minutes: Some(10_080),
                resets_at: Some(1_788_643_388),
            }),
            secondary: None,
            credits: None,
            individual_limit: None,
            spend_control_reached: None,
            plan_type: None,
            rate_limit_reached_type: None,
        }),
    }))
}

fn user_message(message: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: message.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn contextual_user_message() -> RolloutItem {
    user_message("<environment_context>context only</environment_context>")
}

fn legacy_user_message_event(message: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
        message: message.to_string(),
        ..Default::default()
    }))
}

fn completed_user_message(turn_id: &str, message: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::ItemCompleted(ItemCompletedEvent {
        thread_id: codex_protocol::ThreadId::from_string("00000000-0000-0000-0000-000000000000")
            .expect("fixture thread id"),
        turn_id: turn_id.to_string(),
        item: TurnItem::UserMessage(UserMessageItem {
            id: format!("user-{turn_id}"),
            client_id: None,
            content: vec![UserInput::Text {
                text: message.to_string(),
                text_elements: Vec::new(),
            }],
        }),
        started_at_ms: Some(0),
        completed_at_ms: 0,
    }))
}

fn agent_message(message: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::AgentMessage {
            id: None,
            author: "worker".to_string(),
            recipient: "root".to_string(),
            content: vec![AgentMessageInputContent::InputText {
                text: message.to_string(),
            }],
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn turn_context(root: &Path, turn_id: &str) -> RolloutItem {
    RolloutItem::TurnContext(TurnContextItem {
        turn_id: Some(turn_id.to_string()),
        cwd: serde_json::from_value(serde_json::json!(root)).expect("absolute cwd"),
        workspace_roots: None,
        current_date: None,
        timezone: None,
        approval_policy: AskForApproval::Never,
        approvals_reviewer: None,
        sandbox_policy: SandboxPolicy::new_read_only_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: "test-model".to_string(),
        comp_hash: None,
        personality: None,
        collaboration_mode: None,
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: None,
        cyber_access_program: None,
        effort: None,
        summary: ReasoningSummary::Auto,
    })
}

fn compacted(message: &str, replacement_history: Option<Vec<ResponseItem>>) -> RolloutItem {
    RolloutItem::Compacted(CompactedItem {
        message: message.to_string(),
        replacement_history: replacement_history
            .map(|items| items.into_iter().map(Into::into).collect()),
        mcp_resource_origins: None,
        window_number: Some(1),
        first_window_id: None,
        previous_window_id: None,
        window_id: None,
    })
}
