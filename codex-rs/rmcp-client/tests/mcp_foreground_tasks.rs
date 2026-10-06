use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use codex_config::types::AuthKeyringBackendKind;
use codex_config::types::OAuthCredentialsStoreMode;
use codex_exec_server::Environment;
use codex_rmcp_client::ElicitationAction;
use codex_rmcp_client::ElicitationResponse;
use codex_rmcp_client::McpProtocolMode;
use codex_rmcp_client::RmcpClient;
use futures::FutureExt;
use pretty_assertions::assert_eq;
use rmcp::model::CallToolResult;
use rmcp::model::ClientCapabilities;
use rmcp::model::ContentBlock;
use rmcp::model::Implementation;
use rmcp::model::InitializeRequestParams;
use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;

const CREATED: &str = "2026-10-06T10:00:00Z";

fn task(status: &str) -> Value {
    json!({"taskId":"review-1", "status":status, "createdAt":CREATED,
        "lastUpdatedAt":CREATED, "ttlMs":null, "pollIntervalMs":20})
}

async fn fixture(
    mode: &'static str,
) -> anyhow::Result<(MockServer, Arc<RmcpClient>, Arc<Mutex<Vec<Value>>>)> {
    let server = MockServer::start().await;
    let calls = Arc::new(Mutex::new(Vec::<Value>::new()));
    let recorded = Arc::clone(&calls);
    let started = Mutex::new(None::<std::time::Instant>);
    Mock::given(method("POST")).respond_with(move |request: &Request| {
        let Ok(body) = request.body_json::<Value>() else { panic!("expected a JSON-RPC request"); };
        let Ok(mut calls) = recorded.lock() else { panic!("request evidence lock poisoned"); };
        calls.push(body.clone());
        let cancelled = calls.iter().any(|r| r["method"] == "tasks/cancel");
        let mut delay = Duration::ZERO;
        let Some(method) = body["method"].as_str() else { panic!("JSON-RPC method missing"); };
        let result = match method {
            "server/discover" => json!({
                "resultType":"complete", "supportedVersions":["2026-07-28"],
                "capabilities":{"tools":{}, "extensions":{"io.modelcontextprotocol/tasks":{}}},
                "_meta":{"io.modelcontextprotocol/serverInfo":{"name":"review-task-test","version":"1"}},
                "ttlMs":0,"cacheScope":"private"
            }),
            "tools/call" => {
                let Ok(mut task_start) = started.lock() else { panic!("task start lock poisoned"); };
                *task_start = Some(std::time::Instant::now());
                let mut result = task("working");
                result["resultType"] = json!("task");
                result
            }
            "tasks/cancel" | "tasks/update" => json!({"resultType":"complete"}),
            "tasks/get" => {
                assert_eq!(body["params"]["taskId"], "review-1");
                let Ok(task_start) = started.lock() else { panic!("task start lock poisoned"); };
                let Some(task_start) = *task_start else { panic!("tools/call must precede tasks/get"); };
                let complete = matches!(mode, "complete" | "input_repeated") && task_start.elapsed() >= Duration::from_millis(300);
                let mut result = task(if cancelled {"cancelled"} else if complete {"completed"} else {"working"});
                result["resultType"] = json!("complete");
                if complete {
                    result["result"] = json!({"resultType":"complete","isError":false,"content":[{"type":"text","text":"revision changed"}]});
                }
                if mode == "mismatch" && !cancelled { result["taskId"] = json!("foreign-review"); }
                if (mode == "reused_generation" || (mode == "reuse_after_abort" && calls.iter().filter(|r| r["method"] == "tasks/get").count() > 1)) && !cancelled {
                    result["createdAt"] = json!("2026-10-06T11:00:00Z");
                    result["lastUpdatedAt"] = result["createdAt"].clone();
                }
                if mode == "zero_interval" && !cancelled { result["pollIntervalMs"] = json!(0); }
                if mode == "stalled" && !cancelled { delay = Duration::from_millis(500); }
                if mode == "input_repeated" && !cancelled && !complete {
                    result["status"] = json!("input_required");
                    result["inputRequests"] = json!({"approval-1":{"method":"elicitation/create","params":{
                        "mode":"form","message":"Approve exact review","requestedSchema":{"type":"object","properties":{"approved":{"type":"boolean"}}}
                    }}});
                }
                result
            }
            other => panic!("unexpected request {other}"),
        };
        ResponseTemplate::new(200)
            .set_body_json(json!({"jsonrpc":"2.0","id":body["id"],"result":result}))
            .set_delay(delay)
    }).mount(&server).await;
    let client = Arc::new(
        RmcpClient::new_streamable_http_client_with_protocol_mode(
            "review-task-test",
            &server.uri(),
            /*bearer_token*/ None,
            /*http_headers*/ None,
            /*env_http_headers*/ None,
            OAuthCredentialsStoreMode::File,
            AuthKeyringBackendKind::default(),
            Environment::default_for_tests().get_http_client(),
            /*auth_provider*/ None,
            McpProtocolMode::V20260728,
        )
        .await?,
    );
    client
        .initialize(
            InitializeRequestParams::new(
                ClientCapabilities::default(),
                Implementation::new("task-test", "1"),
            ),
            Some(Duration::from_secs(2)),
            Box::new(move |_, _| {
                async move {
                    if mode != "input_repeated" {
                        anyhow::bail!("no model or human input expected");
                    }
                    Ok(ElicitationResponse {
                        action: ElicitationAction::Accept,
                        content: Some(json!({"approved":true})),
                        meta: None,
                    })
                }
                .boxed()
            }),
        )
        .await?;
    Ok((server, client, calls))
}

async fn call(client: &RmcpClient) -> anyhow::Result<CallToolResult> {
    client
        .call_tool(
            "review_wait".to_owned(),
            Some(json!({"cursor":7})),
            /*meta*/ None,
            Some(Duration::from_millis(100)),
        )
        .await
}

#[tokio::test]
async fn a_live_foreground_task_outlives_the_request_deadline_without_recalling_the_tool()
-> anyhow::Result<()> {
    let (_server, client, calls) = fixture("complete").await?;
    let started = std::time::Instant::now();
    assert_eq!(
        call(&client).await?,
        CallToolResult::success(vec![ContentBlock::text("revision changed")])
    );
    assert!(started.elapsed() >= Duration::from_millis(200));
    let methods = calls
        .lock()
        .expect("evidence")
        .iter()
        .map(|r| r["method"].as_str().expect("method").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(methods.iter().filter(|m| *m == "tools/call").count(), 1);
    assert!(!methods.iter().any(|m| m == "tasks/cancel"));
    assert_eq!(calls.lock().expect("evidence").iter().find(|r| r["method"] == "tools/call")
        .and_then(|r| r.pointer("/params/_meta/io.modelcontextprotocol~1clientCapabilities/extensions/io.modelcontextprotocol~1tasks")).cloned(), Some(json!({})));
    client.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn dropping_the_native_tool_future_cancels_only_its_task_on_the_same_connection()
-> anyhow::Result<()> {
    let (_server, client, calls) = fixture("waiting").await?;
    let called = Arc::clone(&client);
    let pending = tokio::spawn(async move { call(&called).await });
    // The outer limit protects the local test's I/O evidence, not the review.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if calls
                .lock()
                .expect("evidence")
                .iter()
                .any(|r| r["method"] == "tasks/get")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    pending.abort();
    assert!(
        pending
            .await
            .expect_err("aborted native future")
            .is_cancelled()
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let methods = calls
                .lock()
                .expect("evidence")
                .iter()
                .map(|r| r["method"].as_str().expect("method").to_owned())
                .collect::<Vec<_>>();
            if methods
                .windows(2)
                .any(|m| m == ["tasks/cancel", "tasks/get"])
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let evidence = calls.lock().expect("evidence").clone();
    assert_eq!(
        evidence
            .iter()
            .filter(|r| r["method"] == "tasks/cancel")
            .count(),
        1
    );
    assert_eq!(
        evidence
            .iter()
            .filter(|r| r["method"] == "tools/call")
            .count(),
        1
    );
    assert!(
        evidence
            .iter()
            .filter(|r| r["method"] == "tasks/cancel")
            .all(|r| r["params"]["taskId"] == "review-1")
    );
    client.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn task_identity_change_is_loud_and_never_restarts_the_original_call() -> anyhow::Result<()> {
    for mode in ["mismatch", "reused_generation"] {
        let (_server, client, calls) = fixture(mode).await?;
        let error = client
            .call_tool(
                "review_wait".to_owned(),
                Some(json!({"cursor":7})),
                /*meta*/ None,
                Some(Duration::from_secs(2)),
            )
            .await
            .expect_err("foreign task");
        assert!(
            error.to_string().contains("identity changed"),
            "unexpected error: {error}"
        );
        assert_eq!(
            calls
                .lock()
                .expect("evidence")
                .iter()
                .filter(|r| r["method"] == "tools/call")
                .count(),
            1
        );
        client.shutdown().await;
    }
    Ok(())
}

#[tokio::test]
async fn a_nonresponding_task_request_has_its_own_io_deadline() -> anyhow::Result<()> {
    let (_server, client, calls) = fixture("stalled").await?;
    let started = std::time::Instant::now();
    assert!(
        call(&client)
            .await
            .expect_err("stalled I/O")
            .to_string()
            .contains("tasks/get I/O timed out")
    );
    assert!(started.elapsed() < Duration::from_millis(400));
    assert_eq!(
        calls
            .lock()
            .expect("evidence")
            .iter()
            .filter(|r| r["method"] == "tools/call")
            .count(),
        1
    );
    client.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn a_zero_task_poll_interval_is_rejected_instead_of_spinning() -> anyhow::Result<()> {
    let (_server, client, _calls) = fixture("zero_interval").await?;
    assert!(
        call(&client)
            .await
            .expect_err("zero interval")
            .to_string()
            .contains("pollIntervalMs must be positive")
    );
    client.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn eventual_task_input_acknowledgement_never_repeats_the_input_submission()
-> anyhow::Result<()> {
    let (_server, client, calls) = fixture("input_repeated").await?;
    assert_eq!(
        call(&client).await?,
        CallToolResult::success(vec![ContentBlock::text("revision changed")])
    );
    let evidence = calls.lock().expect("evidence").clone();
    let updates = evidence
        .iter()
        .filter(|r| r["method"] == "tasks/update")
        .collect::<Vec<_>>();
    assert_eq!(updates.len(), 1);
    assert_eq!(
        updates[0].pointer("/params/inputResponses/approval-1/action"),
        Some(&json!("accept"))
    );
    client.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn interruption_never_cancels_a_reused_task_generation() -> anyhow::Result<()> {
    let (_server, client, calls) = fixture("reuse_after_abort").await?;
    let called = Arc::clone(&client);
    let pending = tokio::spawn(async move { call(&called).await });
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if calls
                .lock()
                .expect("evidence")
                .iter()
                .any(|r| r["method"] == "tasks/get")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    pending.abort();
    assert!(
        pending
            .await
            .expect_err("aborted native future")
            .is_cancelled()
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if calls
                .lock()
                .expect("evidence")
                .iter()
                .filter(|r| r["method"] == "tasks/get")
                .count()
                >= 2
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !calls
            .lock()
            .expect("evidence")
            .iter()
            .any(|r| r["method"] == "tasks/cancel")
    );
    client.shutdown().await;
    Ok(())
}
