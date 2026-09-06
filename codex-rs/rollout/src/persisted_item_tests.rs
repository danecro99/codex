use std::path::Path;
use std::path::PathBuf;

use codex_protocol::ResponseItemId;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::ThreadHistoryMode;
use pretty_assertions::assert_eq;
use serde_json::Value;
use tempfile::TempDir;

use crate::RolloutItem;
use crate::RolloutLine;
use crate::append_rollout_item_to_path;
use crate::decode_rollout_line;
use crate::persisted_rollout_item;

fn response_item(item: ResponseItem) -> RolloutItem {
    RolloutItem::ResponseItem(item.into())
}

fn reasoning(content: Option<Vec<ReasoningItemContent>>) -> RolloutItem {
    response_item(ResponseItem::Reasoning {
        id: Some(ResponseItemId::with_suffix("rs", "1")),
        summary: vec![ReasoningItemReasoningSummary::SummaryText {
            text: "summary".to_string(),
        }],
        content,
        encrypted_content: Some("encrypted".to_string()),
        internal_chat_message_metadata_passthrough: None,
    })
}

fn message(role: &str, content: ContentItem) -> RolloutItem {
    response_item(ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![content],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    })
}

fn function_call() -> RolloutItem {
    response_item(ResponseItem::FunctionCall {
        id: None,
        name: "shell".to_string(),
        namespace: None,
        arguments: r#"{"command":["echo","hi"]}"#.to_string(),
        encrypted_function_args: None,
        call_id: "call-1".to_string(),
        internal_chat_message_metadata_passthrough: None,
    })
}

fn function_call_output() -> RolloutItem {
    response_item(ResponseItem::FunctionCallOutput {
        id: None,
        call_id: Some("call-1".to_string()),
        name: None,
        namespace: None,
        output: FunctionCallOutputPayload {
            body: FunctionCallOutputBody::Text("hi\n".to_string()),
            success: Some(true),
        },
        internal_chat_message_metadata_passthrough: None,
    })
}

fn turn_items() -> Vec<RolloutItem> {
    vec![
        message(
            "user",
            ContentItem::InputText {
                text: "hello".to_string(),
            },
        ),
        reasoning(Some(vec![ReasoningItemContent::Text {
            text: "raw reasoning".to_string(),
        }])),
        reasoning(Some(Vec::new())),
        reasoning(Some(vec![ReasoningItemContent::ReasoningText {
            text: "visible reasoning".to_string(),
        }])),
        reasoning(None),
        function_call(),
        function_call_output(),
        message(
            "assistant",
            ContentItem::OutputText {
                text: "done".to_string(),
            },
        ),
    ]
}

fn seed_paginated_rollout(home: &Path) -> PathBuf {
    let thread_id = ThreadId::new();
    let path = home.join(format!("rollout-2026-09-06T00-00-00-{thread_id}.jsonl"));
    let session_meta = RolloutLine {
        timestamp: "2026-09-06T00:00:00.000Z".to_string(),
        ordinal: Some(0),
        item: RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                session_id: thread_id.into(),
                id: thread_id,
                history_mode: ThreadHistoryMode::Paginated,
                ..SessionMeta::default()
            },
            git: None,
        }),
    };
    let mut encoded = serde_json::to_string(&session_meta).expect("encode session meta");
    encoded.push('\n');
    std::fs::write(path.as_path(), encoded).expect("seed rollout");
    path
}

fn json(item: &RolloutItem) -> Value {
    serde_json::to_value(item).expect("serialize rollout item")
}

fn payload_field<'a>(item: &'a Value, field: &str) -> Option<&'a Value> {
    item.get("payload")?.get(field)
}

/// The persisted encoding is deliberately lossy, so a record can never reproduce every in-memory
/// field. A producer that fingerprints a write intent must therefore describe the persisted form.
#[tokio::test]
async fn durable_records_decode_to_the_persisted_form_of_every_turn_item() {
    let home = TempDir::new().expect("temp dir");
    let path = seed_paginated_rollout(home.path());
    let items = turn_items();
    for item in &items {
        append_rollout_item_to_path(path.as_path(), item)
            .await
            .expect("append rollout item");
    }

    let transcript = std::fs::read_to_string(path.as_path()).expect("read rollout");
    let decoded = transcript
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            decode_rollout_line(serde_json::from_str(line).expect("record is valid json"))
                .expect("record decodes")
        })
        .collect::<Vec<_>>();

    assert_eq!(decoded.len(), items.len() + 1);
    assert_eq!(
        decoded.iter().map(|line| line.ordinal).collect::<Vec<_>>(),
        (0..=u64::try_from(items.len()).expect("ordinal"))
            .map(Some)
            .collect::<Vec<_>>()
    );
    for (item, line) in items.iter().zip(decoded.iter().skip(1)) {
        assert_eq!(
            json(&line.item),
            json(&persisted_rollout_item(item).expect("normalize written item"))
        );
    }
}

/// Reasoning whose content carries no `reasoning_text` is dropped on the way to disk and decodes
/// back as `None`, which serializes again as an explicit `null`. Fingerprinting the raw item would
/// therefore reject the record that faithfully carried it.
#[test]
fn reasoning_without_reasoning_text_is_not_a_serialization_fixed_point() {
    for content in [
        Some(vec![ReasoningItemContent::Text {
            text: "raw reasoning".to_string(),
        }]),
        Some(Vec::new()),
    ] {
        let item = reasoning(content);
        let persisted = persisted_rollout_item(&item).expect("normalize reasoning");
        assert_ne!(json(&item), json(&persisted));
        assert_eq!(payload_field(&json(&item), "content"), None);
        assert_eq!(
            payload_field(&json(&persisted), "content"),
            Some(&Value::Null)
        );
    }
}

/// Normalizing once is enough: the persisted form is stable under further round trips, so both
/// sides of a durability fingerprint converge on the same bytes.
#[test]
fn the_persisted_form_is_stable_under_further_round_trips() {
    for item in turn_items() {
        let persisted = persisted_rollout_item(&item).expect("normalize item");
        let twice = persisted_rollout_item(&persisted).expect("normalize persisted item");
        assert_eq!(json(&persisted), json(&twice));
    }
}
