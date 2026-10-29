use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use app_test_support::MockResponsesConfig;
use app_test_support::TestAppServer;
use app_test_support::create_mock_responses_server_sequence_unchecked;
use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::routing::get;
use axum::routing::post;
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::ListMcpServerStatusParams;
use codex_app_server_protocol::ListMcpServerStatusResponse;
use codex_app_server_protocol::McpServerConnectionStatus;
use codex_app_server_protocol::McpServerOauthLoginCompletedNotification;
use codex_app_server_protocol::McpServerOauthLoginResponse;
use codex_app_server_protocol::McpServerStatusDetail;
use codex_app_server_protocol::PluginInstallParams;
use codex_app_server_protocol::PluginInstallResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
use codex_app_server_protocol::UserInput;
use codex_core::config::set_project_trust_level;
use codex_http_client::HttpClientBuilder;
use codex_protocol::config_types::TrustLevel;
use codex_rmcp_client::McpOAuthCallbackMode;
use codex_rmcp_client::resolve_mcp_oauth_callback_url;
use codex_utils_absolute_path::AbsolutePathBuf;
use core_test_support::responses;
use core_test_support::skip_if_remote;
use core_test_support::stdio_server_bin;
use pretty_assertions::assert_eq;
use rmcp::handler::server::ServerHandler;
use rmcp::model::CallToolRequestParams;
use rmcp::model::CallToolResult;
use rmcp::model::Implementation;
use rmcp::model::JsonObject;
use rmcp::model::ListResourceTemplatesResult;
use rmcp::model::ListResourcesResult;
use rmcp::model::ListToolsResult;
use rmcp::model::PaginatedRequestParams;
use rmcp::model::ServerCapabilities;
use rmcp::model::ServerConfig;
use rmcp::model::Tool;
use rmcp::model::ToolAnnotations;
use rmcp::service::RequestContext;
use rmcp::service::RoleServer;
use rmcp::transport::StreamableHttpServerConfig;
use rmcp::transport::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use serde_json::json;
use tempfile::TempDir;
use test_case::test_case;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio::time::timeout;
use url::Url;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::header;
use wiremock::matchers::method;
use wiremock::matchers::path;

const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Explicit reload must replace catalogs and clients together, including when config is unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reload_unchanged_config_rebinds_threads_without_replaying_active_calls() -> Result<()> {
    struct Catalog {
        revision: AtomicUsize,
        next_client: AtomicUsize,
        listed: mpsc::UnboundedSender<(usize, usize)>,
        started: mpsc::UnboundedSender<serde_json::Value>,
        completed: mpsc::UnboundedSender<serde_json::Value>,
        release: tokio::sync::Notify,
    }
    struct CatalogServer {
        catalog: Arc<Catalog>,
        client: usize,
        listed_revision: AtomicUsize,
    }
    impl ServerHandler for CatalogServer {
        fn get_info(&self) -> ServerConfig {
            ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, rmcp::ErrorData> {
            let revision = self.catalog.revision.load(Ordering::SeqCst);
            self.listed_revision.store(revision, Ordering::SeqCst);
            self.catalog.listed.send((self.client, revision)).unwrap();
            let schema: JsonObject = serde_json::from_value(json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": [format!("schema-{revision}")]},
                    "hold": {"type": "boolean"}
                },
                "required": ["value", "hold"],
                "additionalProperties": false
            }))
            .unwrap();
            let mut tool = Tool::new(
                "write_probe",
                format!("catalog-{revision}-client-{}", self.client),
                Arc::new(schema),
            );
            tool.annotations = Some(ToolAnnotations::new().read_only(false));
            Ok(ListToolsResult::with_all_items(vec![tool]))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
            let revision = self.listed_revision.load(Ordering::SeqCst);
            let arguments = request.arguments.unwrap();
            // The fixture keeps the descriptor fence: a new schema on an old client is an error.
            if arguments["value"] != format!("schema-{revision}") {
                return Err(rmcp::ErrorData::invalid_params(
                    "schema/client mismatch",
                    None,
                ));
            }
            let call = json!({"client": self.client, "revision": revision, "arguments": arguments});
            self.catalog.started.send(call.clone()).unwrap();
            if call["arguments"]["hold"] == true {
                self.catalog.release.notified().await;
            }
            self.catalog.completed.send(call.clone()).unwrap();
            Ok(CallToolResult::structured(call).into())
        }
    }

    async fn bound_tool(app: &mut TestAppServer, thread: &str) -> Result<serde_json::Value> {
        let response: ListMcpServerStatusResponse = app
            .request(|request_id| ClientRequest::McpServerStatusList {
                request_id,
                params: ListMcpServerStatusParams {
                    thread_id: Some(thread.to_string()),
                    server_name: Some("reload_probe".to_string()),
                    detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                    cursor: None,
                    limit: None,
                },
            })
            .await?;
        assert_eq!(response.next_cursor, None);
        assert_eq!(response.data.len(), 1);
        Ok(serde_json::to_value(
            &response.data[0].tools["write_probe"],
        )?)
    }

    let (listed, mut lists) = mpsc::unbounded_channel();
    let (started, mut starts) = mpsc::unbounded_channel();
    let (completed, mut completions) = mpsc::unbounded_channel();
    let catalog = Arc::new(Catalog {
        revision: AtomicUsize::new(0),
        next_client: AtomicUsize::new(0),
        listed,
        started,
        completed,
        release: tokio::sync::Notify::new(),
    });
    let service_catalog = Arc::clone(&catalog);
    let service = StreamableHttpService::new(
        move || {
            Ok(CatalogServer {
                client: service_catalog.next_client.fetch_add(1, Ordering::SeqCst),
                catalog: Arc::clone(&service_catalog),
                listed_revision: AtomicUsize::new(usize::MAX),
            })
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let http_server = tokio::spawn(async move {
        axum::serve(listener, Router::new().nest_service("/mcp", service))
            .await
            .unwrap();
    });
    let model = responses::start_mock_server().await;
    let home = TempDir::new()?;
    MockResponsesConfig::new(&model.uri())
        .with_extra_config(&format!(
            "[mcp_servers.reload_probe]\nurl = \"http://{address}/mcp\"\nrequired = true\n\
             [mcp_servers.reload_probe.tools.write_probe]\napproval_mode = \"approve\"\n"
        ))
        .write(home.path())?;
    let original_config = std::fs::read(home.path().join("config.toml"))?;
    let mut app = TestAppServer::builder()
        .with_codex_home(home.path())
        .build_initialized()
        .await?;
    let first = app
        .start_thread(ThreadStartParams::default())
        .await?
        .thread
        .id;
    let second = app
        .start_thread(ThreadStartParams::default())
        .await?
        .thread
        .id;
    let old_tools = [
        bound_tool(&mut app, &first).await?,
        bound_tool(&mut app, &second).await?,
    ];
    let mut old_clients = Vec::new();
    for _ in 0..2 {
        let (client, revision) = timeout(DEFAULT_READ_TIMEOUT, lists.recv())
            .await?
            .context("initial tools/list")?;
        assert_eq!(revision, 0);
        old_clients.push(client);
    }

    let namespace = "mcp__reload_probe";
    let mock = responses::mount_sse_sequence(
        &model,
        vec![
            responses::sse(vec![
                responses::ev_response_created("held"),
                responses::ev_function_call_with_namespace(
                    "held-call",
                    namespace,
                    "write_probe",
                    &json!({"value": "schema-0", "hold": true}).to_string(),
                ),
                responses::ev_completed("held"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("fresh-first"),
                responses::ev_function_call_with_namespace(
                    "fresh-first-call",
                    namespace,
                    "write_probe",
                    &json!({"value": "schema-1", "hold": false}).to_string(),
                ),
                responses::ev_completed("fresh-first"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("done-first"),
                responses::ev_assistant_message("first-done", "done"),
                responses::ev_completed("done-first"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("fresh-second"),
                responses::ev_function_call_with_namespace(
                    "fresh-second-call",
                    namespace,
                    "write_probe",
                    &json!({"value": "schema-1", "hold": false}).to_string(),
                ),
                responses::ev_completed("fresh-second"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("done-second"),
                responses::ev_assistant_message("second-done", "done"),
                responses::ev_completed("done-second"),
            ]),
        ],
    )
    .await;
    let request = app
        .send_turn_start_request(TurnStartParams {
            thread_id: first.clone(),
            input: vec![UserInput::Text {
                text: "Call the write probe".to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let _: TurnStartResponse = timeout(DEFAULT_READ_TIMEOUT, app.read_response(request)).await??;
    let held = timeout(DEFAULT_READ_TIMEOUT, starts.recv())
        .await?
        .context("held call started")?;
    assert_eq!(held["revision"], 0);
    assert_eq!(
        held["arguments"],
        json!({"value": "schema-0", "hold": true})
    );

    catalog.revision.store(1, Ordering::SeqCst);
    let reload = app
        .send_raw_request("config/mcpServer/reload", None)
        .await?;
    let ack: serde_json::Value = timeout(DEFAULT_READ_TIMEOUT, app.read_response(reload)).await??;
    assert_eq!(ack, json!({}));
    let mut new_clients = Vec::new();
    for _ in 0..2 {
        let (client, revision) = timeout(DEFAULT_READ_TIMEOUT, lists.recv())
            .await
            .expect(
                "explicit reload must perform fresh tools/list for both unchanged thread configs",
            )
            .context("fresh tools/list")?;
        assert_eq!(revision, 1);
        assert!(!old_clients.contains(&client));
        new_clients.push(client);
    }
    assert_ne!(new_clients[0], new_clients[1]);
    let fresh_tools = [
        bound_tool(&mut app, &first).await?,
        bound_tool(&mut app, &second).await?,
    ];
    assert!(
        completions.try_recv().is_err(),
        "reload must not terminate the held write"
    );
    assert_eq!(
        std::fs::read(home.path().join("config.toml"))?,
        original_config
    );
    catalog.release.notify_one();
    let held_completed = timeout(DEFAULT_READ_TIMEOUT, completions.recv())
        .await?
        .context("held call completed")?;
    assert_eq!(held_completed, held);
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_notification_message("turn/completed"),
    )
    .await??;

    let request = app
        .send_turn_start_request(TurnStartParams {
            thread_id: second.clone(),
            input: vec![UserInput::Text {
                text: "Use the refreshed write probe".to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let _: TurnStartResponse = timeout(DEFAULT_READ_TIMEOUT, app.read_response(request)).await??;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_notification_message("turn/completed"),
    )
    .await??;
    let requests = mock.requests();
    assert_eq!(requests.len(), 5);
    let held_output = requests[1]
        .function_call_output_text("held-call")
        .context("original held structured result must reach the model")?;
    let (_, held_body) = held_output
        .split_once("\nOutput:\n")
        .context("MCP output includes its wall-time header")?;
    let original_result: serde_json::Value = serde_json::from_str(held_body)?;
    assert_eq!(original_result, held);
    for (request, expected) in requests.iter().zip([
        &old_tools[0],
        &fresh_tools[0],
        &fresh_tools[0],
        &fresh_tools[1],
        &fresh_tools[1],
    ]) {
        let tool = request
            .tool_by_name(namespace, "write_probe")
            .context("model-visible write probe")?;
        assert_eq!(
            json!({"description": tool["description"], "schema": tool["parameters"]}),
            json!({"description": expected["description"], "schema": expected["inputSchema"]})
        );
    }
    for expected in &fresh_tools {
        let call = timeout(DEFAULT_READ_TIMEOUT, starts.recv())
            .await?
            .context("fresh model call")?;
        assert_eq!(call["revision"], 1);
        assert_eq!(
            call["arguments"],
            json!({"value": "schema-1", "hold": false})
        );
        assert_eq!(
            expected["description"],
            format!("catalog-1-client-{}", call["client"])
        );
        let completed = timeout(DEFAULT_READ_TIMEOUT, completions.recv())
            .await?
            .context("fresh call completion")?;
        assert_eq!(completed, call);
    }
    assert!(starts.try_recv().is_err(), "no call replay");
    assert!(
        completions.try_recv().is_err(),
        "exactly one completion per call"
    );
    assert!(
        lists.try_recv().is_err(),
        "one tools/list per new thread client"
    );
    for thread in [&first, &second] {
        let read = app
            .send_raw_request("thread/read", Some(json!({"threadId": thread})))
            .await?;
        let response: serde_json::Value =
            timeout(DEFAULT_READ_TIMEOUT, app.read_response(read)).await??;
        assert_eq!(response["thread"]["id"], *thread);
    }
    http_server.abort();
    Ok(())
}

/// Selected status reads return the thread's tools without repeating MCP discovery.
#[tokio::test]
async fn mcp_server_status_list_reuses_thread_connection() -> Result<()> {
    let server = MockServer::start().await;
    let tool = json!({
        "name": "open_widget",
        "title": "Open widget",
        "inputSchema": { "type": "object" },
        "_meta": { "ui": { "resourceUri": "ui://widget.html" } }
    });
    let response_tool = tool.clone();
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(move |request: &wiremock::Request| {
            let request: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            let result = match request["method"].as_str() {
                Some("initialize") => json!({
                    "protocolVersion": "2025-11-25",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "widget", "version": "1.0.0" }
                }),
                Some("tools/list") => json!({ "tools": [response_tool.clone()] }),
                Some("resources/list") => json!({ "resources": [] }),
                Some("resources/templates/list") => json!({ "resourceTemplates": [] }),
                _ => return ResponseTemplate::new(202),
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": request["id"], "result": result
            }))
        })
        .mount(&server)
        .await;
    let codex_home = TempDir::new()?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "[mcp_servers.widget]\nurl = \"{}/mcp\"",
            server.uri()
        ))
        .write(codex_home.path())?;
    let mut app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let thread = app.start_thread(ThreadStartParams::default()).await?.thread;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_matching_notification("widget MCP ready", |notification| {
            notification.method == "mcpServer/startupStatus/updated"
                && notification.params.as_ref().is_some_and(|params| {
                    params["threadId"] == thread.id
                        && params["name"] == "widget"
                        && params["status"] == "ready"
                })
        }),
    )
    .await??;

    for detail in [
        McpServerStatusDetail::ToolsAndAuthOnly,
        McpServerStatusDetail::Full,
    ] {
        let response: ListMcpServerStatusResponse = app
            .request(|request_id| ClientRequest::McpServerStatusList {
                request_id,
                params: ListMcpServerStatusParams {
                    thread_id: Some(thread.id.clone()),
                    server_name: Some("widget".to_string()),
                    detail: Some(detail),
                    cursor: None,
                    limit: None,
                },
            })
            .await?;
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].name, "widget");
        assert_eq!(
            response.data[0].tools,
            std::collections::HashMap::from([(
                "open_widget".to_string(),
                serde_json::from_value(tool.clone())?
            )])
        );
        assert_eq!(response.next_cursor, None);
    }
    let discovery_methods: Vec<_> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter_map(|request| serde_json::from_slice::<serde_json::Value>(&request.body).ok())
        .filter_map(|request| request["method"].as_str().map(str::to_owned))
        .filter(|method| matches!(method.as_str(), "initialize" | "tools/list"))
        .collect();
    assert_eq!(discovery_methods, ["initialize", "tools/list"]);
    Ok(())
}

#[test_case(false, None, None, None, None, true; "legacy callback")]
#[test_case(
    false,
    Some("http://127.0.0.1/callback/registered"),
    None,
    None,
    None,
    true;
    "ordinary configured callback falls back to default"
)]
#[test_case(
    false,
    Some("http://127.0.0.1/callback/registered"),
    Some("http://127.0.0.1/global/callback"),
    None,
    None,
    true;
    "ordinary configured callback falls back to global"
)]
#[test_case(
    false,
    Some("http://127.0.0.1/callback/registered"),
    Some("http://127.0.0.1/global/callback"),
    Some("https://unexpected.example"),
    None,
    false;
    "legacy callback fallback rejects mismatched issuer"
)]
#[test_case(
    true,
    Some("http://127.0.0.1/callback"),
    None,
    Some("matching"),
    None,
    true;
    "matching issuer"
)]
#[test_case(
    true,
    Some("http://127.0.0.1/callback"),
    None,
    Some("https://unexpected.example"),
    None,
    false;
    "mismatched issuer"
)]
#[test_case(
    true,
    Some("http://127.0.0.1/callback"),
    None,
    None,
    None,
    false;
    "missing issuer"
)]
#[test_case(
    false,
    None,
    None,
    None,
    Some("test-client-secret"),
    true;
    "client secret with legacy callback"
)]
#[test_case(
    true,
    Some("http://127.0.0.1/callback"),
    None,
    Some("matching"),
    Some("test-client-secret"),
    true;
    "client secret with matching issuer"
)]
#[tokio::test]
async fn oauth_login_validates_callback_issuer_and_uses_http_headers_helper(
    issuer_supported: bool,
    configured_callback: Option<&str>,
    global_callback: Option<&str>,
    callback_issuer: Option<&str>,
    client_secret: Option<&'static str>,
    expected_success: bool,
) -> Result<()> {
    let oauth = MockServer::start().await;
    let issuer = format!("{}/mcp", oauth.uri());
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server/mcp"))
        .and(header("x-gateway", "gateway-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "issuer": issuer,
            "authorization_endpoint": format!("{}/oauth/authorize", oauth.uri()),
            "token_endpoint": format!("{}/oauth/token", oauth.uri()),
            "authorization_response_iss_parameter_supported": issuer_supported,
            "token_endpoint_auth_methods_supported": [if client_secret.is_some() {
                "client_secret_post"
            } else {
                "none"
            }],
        })))
        .mount(&oauth)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(header("x-gateway", "gateway-token"))
        .and(move |request: &wiremock::Request| {
            let body: BTreeMap<_, _> = url::form_urlencoded::parse(&request.body)
                .into_owned()
                .collect();
            body.get("client_secret").map(String::as_str) == client_secret
                && body.get("client_id").map(String::as_str) == Some("test-client")
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "oauth-token",
            "token_type": "Bearer",
        })))
        .expect(u64::from(expected_success))
        .mount(&oauth)
        .await;

    let codex_home = TempDir::new()?;
    let helper_command = if cfg!(windows) {
        r#"echo {"X-Gateway":"gateway-token"}"#
    } else {
        r#"printf '{"X-Gateway":"gateway-token"}'"#
    };
    let saved_callback = configured_callback
        .map(|callback| format!("callback_url = \"{callback}\"\n"))
        .unwrap_or_default();
    let saved_client_secret = client_secret
        .map(|secret| format!("client_secret = \"{secret}\"\n"))
        .unwrap_or_default();
    let global_callback_config = global_callback
        .map(|callback| format!("mcp_oauth_callback_url = \"{callback}\"\n"))
        .unwrap_or_default();
    std::fs::write(
        codex_home.path().join("config.toml"),
        format!(
            "mcp_oauth_credentials_store = \"file\"\n\
             {global_callback_config}\
             [mcp_servers.gateway]\n\
             url = \"{}/mcp\"\n\
             http_headers_helper = {}\n\
             [mcp_servers.gateway.oauth]\n\
             client_id = \"test-client\"\n\
             {saved_client_secret}\
             {saved_callback}",
            oauth.uri(),
            toml::Value::String(helper_command.to_string()),
        ),
    )?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized_with_timeout(DEFAULT_READ_TIMEOUT)
        .await?;

    let request_id = mcp
        .send_raw_request(
            "mcpServer/oauth/login",
            Some(json!({"name": "gateway", "timeoutSecs": 10})),
        )
        .await?;
    let response: McpServerOauthLoginResponse =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(request_id)).await??;
    let authorization_url = Url::parse(&response.authorization_url)?;
    let query: BTreeMap<_, _> = authorization_url.query_pairs().into_owned().collect();
    assert!(!query.contains_key("client_secret"));
    let mut callback_url = Url::parse(&query["redirect_uri"])?;
    let expected_callback = if issuer_supported {
        configured_callback
            .expect("issuer-bound cases configure a stable callback")
            .to_string()
    } else {
        resolve_mcp_oauth_callback_url(
            &issuer,
            global_callback,
            McpOAuthCallbackMode::CallbackSpecific,
        )?
    };
    assert_eq!(callback_url.path(), Url::parse(&expected_callback)?.path());
    callback_url
        .query_pairs_mut()
        .append_pair("code", "test-code")
        .append_pair("state", &query["state"]);
    if let Some(callback_issuer) = callback_issuer {
        let callback_issuer = if callback_issuer == "matching" {
            issuer.as_str()
        } else {
            callback_issuer
        };
        callback_url
            .query_pairs_mut()
            .append_pair("iss", callback_issuer);
    }
    HttpClientBuilder::new()
        .build_direct()?
        .get(callback_url)
        .send()
        .await?
        .error_for_status()?;
    let completed: McpServerOauthLoginCompletedNotification = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_notification("mcpServer/oauthLogin/completed"),
    )
    .await??;
    assert_eq!(
        completed.login_id.as_deref(),
        Some(
            response
                .login_id
                .as_deref()
                .expect("login response should contain an ID")
        )
    );
    assert_eq!(
        (
            completed.name.as_str(),
            completed.thread_id,
            completed.success,
            completed.error.is_some(),
        ),
        ("gateway", None, expected_success, !expected_success)
    );
    if let Some(client_secret) = client_secret {
        assert!(!response.authorization_url.contains(client_secret));
        let credentials = std::fs::read_to_string(codex_home.path().join(".credentials.json"))?;
        assert!(!credentials.contains(client_secret));
    }
    oauth.verify().await;
    Ok(())
}

#[test_case(false; "plain HTTP")]
#[test_case(true; "HTTP headers helper")]
#[tokio::test]
async fn oauth_login_rejects_servers_disabled_by_managed_requirements(
    with_headers_helper: bool,
) -> Result<()> {
    let codex_home = TempDir::new()?;
    let marker = codex_home.path().join("helper-ran");
    let helper = with_headers_helper
        .then(|| toml::Value::String(format!("echo invoked > \"{}\"", marker.display())))
        .map_or_else(String::new, |command| {
            format!("http_headers_helper = {command}\n")
        });
    std::fs::write(
        codex_home.path().join("config.toml"),
        format!("[mcp_servers.blocked]\nurl = \"https://example.com/mcp\"\n{helper}"),
    )?;
    std::fs::write(
        codex_home.path().join("requirements.toml"),
        "[mcp_servers.blocked.identity]\nurl = \"https://allowed.example.com/mcp\"\n",
    )?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized_with_timeout(DEFAULT_READ_TIMEOUT)
        .await?;

    let request_id = mcp
        .send_raw_request(
            "mcpServer/oauth/login",
            Some(json!({"name": "blocked", "timeoutSecs": 10})),
        )
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;
    assert!(
        error
            .error
            .message
            .contains("disabled by managed requirements")
    );
    assert!(!marker.exists());
    Ok(())
}

async fn wait_for_new_pid(path: &Path, previous_pid: Option<&str>) -> Result<String> {
    Ok(timeout(DEFAULT_READ_TIMEOUT, async {
        loop {
            if let Ok(contents) = std::fs::read_to_string(path) {
                let pid = contents.trim();
                if !pid.is_empty() && Some(pid) != previous_pid {
                    return pid.to_string();
                }
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await?)
}

fn assert_dynamic_status(response: &ListMcpServerStatusResponse, process_label: &str) {
    assert_eq!(response.data.len(), 1);
    let status = &response.data[0];
    assert_eq!(status.name, "cached-stdio");
    assert_eq!(
        status
            .server_info
            .as_ref()
            .and_then(|info| info.title.as_deref()),
        Some(process_label)
    );
    assert_eq!(
        status
            .tools
            .get("echo")
            .and_then(|tool| tool.description.as_deref()),
        Some(format!("Echo from {process_label}.").as_str())
    );
}

#[tokio::test]
async fn oauth_login_automatically_selects_callback_specific_cimd_without_metadata_issuer()
-> Result<()> {
    let responses_server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let metadata = json!({
        "authorization_endpoint": format!("{base_url}/authorize"),
        "token_endpoint": format!("{base_url}/token"),
        "registration_endpoint": format!("{base_url}/register"),
        "client_id_metadata_document_supported": true,
        "token_endpoint_auth_methods_supported": ["none"],
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
    });
    let registrations = Arc::new(AtomicUsize::new(0));
    let registration_count = Arc::clone(&registrations);
    let (token_request_tx, mut token_request_rx) = mpsc::unbounded_channel();
    let (mcp_authorization_tx, mut mcp_authorization_rx) = mpsc::unbounded_channel();
    let tool_name = Arc::new("cimd".to_string());
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(McpStatusServer {
                tool_name: Arc::clone(&tool_name),
                tools_error: None,
            })
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let mcp_router =
        Router::new()
            .nest_service("/mcp", mcp_service)
            .layer(axum::middleware::from_fn(
                move |request: axum::extract::Request, next: axum::middleware::Next| {
                    let mcp_authorization_tx = mcp_authorization_tx.clone();
                    async move {
                        if let Some(authorization) = request
                            .headers()
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                        {
                            let _ = mcp_authorization_tx.send(authorization.to_string());
                        }
                        next.run(request).await
                    }
                },
            ));
    let oauth_server = Router::new()
        .route(
            "/.well-known/oauth-authorization-server/mcp",
            get(move || {
                let metadata = metadata.clone();
                async move { Json(metadata) }
            }),
        )
        .route(
            "/register",
            post(move || {
                let registrations = Arc::clone(&registration_count);
                async move {
                    registrations.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"client_id": "unexpected-dcr-client"}))
                }
            }),
        )
        .route(
            "/token",
            post(move |headers: HeaderMap, body: Bytes| {
                let token_request_tx = token_request_tx.clone();
                async move {
                    let _ = token_request_tx.send((
                        String::from_utf8_lossy(&body).into_owned(),
                        headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_string),
                    ));
                    Json(json!({
                        "access_token": "cimd-access-token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "test-refresh-token",
                    }))
                }
            }),
        )
        .merge(mcp_router);
    let oauth_server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, oauth_server).await;
    });

    let codex_home = TempDir::new()?;
    mock_responses_config(&responses_server.uri())
        .with_extra_config(&format!(
            "mcp_oauth_credentials_store = \"file\"\n[mcp_servers.cimd]\nurl = \"{base_url}/mcp\""
        ))
        .write(codex_home.path())?;
    let mut app_server = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;

    let request_id = app_server
        .send_raw_request(
            "mcpServer/oauth/login",
            Some(json!({"name": "cimd", "timeoutSecs": 10})),
        )
        .await?;
    let response: McpServerOauthLoginResponse =
        timeout(DEFAULT_READ_TIMEOUT, app_server.read_response(request_id)).await??;
    let authorization_url = Url::parse(&response.authorization_url)?;
    let parameters = authorization_url
        .query_pairs()
        .into_owned()
        .collect::<BTreeMap<String, String>>();
    let redirect_uri = parameters["redirect_uri"].clone();
    let mut callback_url = Url::parse(&redirect_uri)?;
    let callback_id = callback_url
        .path()
        .strip_prefix("/callback/")
        .expect("issuerless CIMD should use a resource-specific callback");
    let client_id = format!("https://chatgpt.com/oauth/codex/{callback_id}/client.json");
    assert_eq!(parameters.get("client_id"), Some(&client_id));
    assert_eq!(
        parameters.get("code_challenge_method").map(String::as_str),
        Some("S256")
    );
    assert_eq!(registrations.load(Ordering::SeqCst), 0);

    callback_url
        .query_pairs_mut()
        .append_pair("code", "cimd-authorization-code")
        .append_pair("state", &parameters["state"]);
    HttpClientBuilder::new()
        .build_direct()?
        .get(callback_url)
        .send()
        .await?
        .error_for_status()?;
    let (token_request, token_authorization) =
        timeout(DEFAULT_READ_TIMEOUT, token_request_rx.recv())
            .await?
            .expect("CIMD authorization should exchange its authorization code");
    let token_parameters = url::form_urlencoded::parse(token_request.as_bytes())
        .into_owned()
        .collect::<BTreeMap<String, String>>();
    assert_eq!(token_parameters.get("client_id"), Some(&client_id));
    assert!(token_parameters.contains_key("code_verifier"));
    assert_eq!(token_authorization, None);

    let completed: McpServerOauthLoginCompletedNotification = timeout(
        DEFAULT_READ_TIMEOUT,
        app_server.read_notification("mcpServer/oauthLogin/completed"),
    )
    .await??;
    assert_eq!(
        completed.login_id.as_deref(),
        Some(
            response
                .login_id
                .as_deref()
                .expect("login response should contain an ID")
        )
    );
    assert_eq!(
        completed,
        McpServerOauthLoginCompletedNotification {
            name: "cimd".to_string(),
            thread_id: None,
            login_id: response.login_id,
            success: true,
            error: None,
        }
    );
    assert_eq!(registrations.load(Ordering::SeqCst), 0);

    let request_id = app_server
        .send_raw_request("config/mcpServer/reload", /*params*/ None)
        .await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app_server.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let _: ListMcpServerStatusResponse = app_server
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::Full),
                thread_id: None,
            },
        })
        .await?;
    assert_eq!(
        timeout(DEFAULT_READ_TIMEOUT, mcp_authorization_rx.recv())
            .await?
            .expect("MCP startup should use the access token"),
        "Bearer cimd-access-token"
    );
    assert_eq!(registrations.load(Ordering::SeqCst), 0);

    oauth_server_handle.abort();
    let _ = oauth_server_handle.await;
    Ok(())
}

#[test_case(false; "configured server")]
#[test_case(true; "plugin server")]
#[tokio::test]
async fn mcp_server_status_list_returns_raw_server_and_tool_names(plugin: bool) -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (mcp_server_url, mcp_server_handle) =
        start_mcp_server("look-up.raw", /*tools_error*/ None).await?;
    let codex_home = TempDir::new()?;
    let endpoint = format!("{mcp_server_url}/mcp?secret=not-for-clients");
    let http_config = if plugin {
        "[features]\nplugins = true\n".to_string()
    } else {
        format!("[mcp_servers.some-server]\nurl = {endpoint:?}\n")
    };
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "{http_config}[mcp_servers.broken-server]\ncommand = {}",
            toml::Value::String(
                codex_home
                    .path()
                    .join("missing-mcp-server")
                    .display()
                    .to_string()
            )
        ))
        .write(codex_home.path())?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;
    let plugin_root = TempDir::new()?;
    if plugin {
        std::fs::create_dir_all(plugin_root.path().join(".git"))?;
        std::fs::create_dir_all(plugin_root.path().join(".agents/plugins"))?;
        std::fs::create_dir_all(plugin_root.path().join("sample/.codex-plugin"))?;
        std::fs::write(
            plugin_root.path().join("sample/.codex-plugin/plugin.json"),
            serde_json::to_vec(&json!({"name": "sample"}))?,
        )?;
        std::fs::write(
            plugin_root.path().join("sample/.mcp.json"),
            serde_json::to_vec(
                &json!({"mcpServers": {"some-server": {"type": "http", "url": endpoint}}}),
            )?,
        )?;
        let marketplace_path = plugin_root.path().join(".agents/plugins/marketplace.json");
        std::fs::write(
            &marketplace_path,
            serde_json::to_vec(&json!({
                "name": "debug", "plugins": [{"name": "sample", "source": {"source": "local", "path": "./sample"}}]
            }))?,
        )?;
        let request_id = mcp
            .send_plugin_install_request(PluginInstallParams {
                marketplace_path: Some(AbsolutePathBuf::try_from(marketplace_path)?),
                remote_marketplace_name: None,
                install_attempt_id: None,
                plugin_name: "sample".to_string(),
            })
            .await?;
        let _: PluginInstallResponse =
            timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(request_id)).await??;
    }
    let response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: None,
                thread_id: None,
            },
        })
        .await?;

    assert_eq!(response.next_cursor, None);
    assert_eq!(response.data.len(), 2);
    let failed = response
        .data
        .iter()
        .find(|status| status.name == "broken-server")
        .expect("broken server status");
    assert!(failed.tools.is_empty());
    assert_eq!(failed.server_capabilities, None);
    assert_eq!(failed.http_origin, None);
    assert!(
        failed
            .tools_error
            .as_deref()
            .is_some_and(|error| error.contains("MCP startup failed"))
    );
    let status = response
        .data
        .iter()
        .find(|status| status.name == "some-server")
        .expect("configured server status");
    assert_eq!(
        status
            .server_capabilities
            .as_ref()
            .and_then(|caps| caps.get("extensions")),
        Some(&json!({
            "openai/settings": { "readTool": "settings.read", "updateTool": "settings.update" }
        }))
    );
    assert_eq!(status.tools_error, None);
    assert_eq!(status.name, "some-server");
    assert_eq!(status.runtime_status, None);
    assert_eq!(
        status.plugin_id.as_deref(),
        plugin.then_some("sample@debug")
    );
    assert_eq!(status.http_origin.as_deref(), Some(mcp_server_url.as_str()));
    assert_eq!(
        status.tools.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["look-up.raw".to_string()])
    );
    assert_eq!(
        status
            .tools
            .get("look-up.raw")
            .map(|tool| tool.name.as_str()),
        Some("look-up.raw")
    );
    assert_eq!(
        status
            .server_info
            .as_ref()
            .and_then(|info| info.title.as_deref()),
        Some("Lookup Server")
    );

    mcp_server_handle.abort();
    let _ = mcp_server_handle.await;

    Ok(())
}

#[tokio::test]
async fn mcp_server_status_list_waits_for_live_stdio_metadata_before_using_cached_tools()
-> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let codex_home = TempDir::new()?;
    let barrier_file = codex_home.path().join("allow-initialize");
    let pid_file = codex_home.path().join("mcp.pid");
    std::fs::write(&barrier_file, "ready")?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            r#"[mcp_servers.cached-stdio]
command = {}
enabled_tools = ["echo"]
startup_timeout_sec = 10

[mcp_servers.cached-stdio.env]
MCP_TEST_DYNAMIC_SERVER_METADATA = "1"
MCP_TEST_INITIALIZE_BARRIER_FILE = {}
MCP_TEST_PID_FILE = {}
"#,
            toml::Value::String(stdio_server_bin()?),
            toml::Value::String(barrier_file.to_string_lossy().into_owned()),
            toml::Value::String(pid_file.to_string_lossy().into_owned()),
        ))
        .write(codex_home.path())?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;
    let first_response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                thread_id: None,
            },
        })
        .await?;
    let first_pid = wait_for_new_pid(&pid_file, /*previous_pid*/ None).await?;
    assert_dynamic_status(&first_response, &format!("rmcp-test-process-{first_pid}"));

    std::fs::remove_file(&barrier_file)?;
    let second_request_id = mcp
        .send_list_mcp_server_status_request(ListMcpServerStatusParams {
            server_name: None,
            cursor: None,
            limit: None,
            detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
            thread_id: None,
        })
        .await?;
    let second_pid = wait_for_new_pid(&pid_file, Some(&first_pid)).await?;
    assert!(
        timeout(
            Duration::from_millis(200),
            mcp.read_stream_until_response_message(RequestId::Integer(second_request_id)),
        )
        .await
        .is_err(),
        "status/list should wait for the live stdio server to initialize"
    );

    std::fs::write(&barrier_file, "ready")?;
    let second_response: ListMcpServerStatusResponse =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(second_request_id)).await??;
    assert_dynamic_status(&second_response, &format!("rmcp-test-process-{second_pid}"));

    Ok(())
}

#[tokio::test]
async fn mcp_server_status_list_uses_thread_project_local_config() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (mcp_server_url, mcp_server_handle) =
        start_mcp_server("project_lookup", /*tools_error*/ None).await?;
    let codex_home = TempDir::new()?;
    let workspace = TempDir::new()?;
    mock_responses_config(&server.uri()).write(codex_home.path())?;
    std::fs::create_dir_all(workspace.path().join(".git"))?;
    set_project_trust_level(codex_home.path(), workspace.path(), TrustLevel::Trusted)?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let ThreadStartResponse { thread, .. } = mcp
        .start_thread(ThreadStartParams {
            cwd: Some(workspace.path().to_string_lossy().into_owned()),
            ..Default::default()
        })
        .await?;

    let project_config_dir = workspace.path().join(".codex");
    std::fs::create_dir_all(&project_config_dir)?;
    std::fs::write(
        project_config_dir.join("config.toml"),
        format!(
            r#"
[mcp_servers.project-server]
url = "{mcp_server_url}/mcp"
"#
        ),
    )?;

    let threadless_response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                thread_id: None,
            },
        })
        .await?;
    assert_eq!(threadless_response.data, Vec::new());

    let thread_response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                thread_id: Some(thread.id),
            },
        })
        .await?;

    assert_eq!(thread_response.next_cursor, None);
    assert_eq!(thread_response.data.len(), 1);
    let status = &thread_response.data[0];
    assert_eq!(status.name, "project-server");
    assert_eq!(status.runtime_status, None);
    assert_eq!(
        status.tools.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["project_lookup".to_string()])
    );

    mcp_server_handle.abort();
    let _ = mcp_server_handle.await;

    Ok(())
}

#[tokio::test]
async fn mcp_server_status_list_reports_thread_runtime_connections() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (mcp_server_url, mcp_server_handle) =
        start_mcp_server("lookup", /*tools_error*/ None).await?;
    let codex_home = TempDir::new()?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "[mcp_servers.connected]\nurl = \"{mcp_server_url}/mcp\"\n\
             [mcp_servers.disabled]\nurl = \"{mcp_server_url}/mcp\"\nenabled = false\n"
        ))
        .write(codex_home.path())?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let ThreadStartResponse { thread, .. } = mcp.start_thread(ThreadStartParams::default()).await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_matching_notification("connected MCP server ready", |notification| {
            notification.method == "mcpServer/startupStatus/updated"
                && notification.params.as_ref().is_some_and(|params| {
                    params["name"] == "connected" && params["status"] == "ready"
                })
        }),
    )
    .await??;
    let response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                thread_id: Some(thread.id.clone()),
            },
        })
        .await?;
    assert_eq!(
        response
            .data
            .into_iter()
            .map(|status| (status.name, status.runtime_status))
            .collect::<Vec<_>>(),
        vec![
            (
                "connected".to_string(),
                Some(McpServerConnectionStatus::Connected)
            ),
            (
                "disabled".to_string(),
                Some(McpServerConnectionStatus::Disabled)
            ),
        ]
    );
    // Inventory uses the latest config, but a same-name replacement is not the
    // connection that this thread started. Unchanged registrations retain status.
    let (replacement_url, replacement_handle) =
        start_mcp_server("replacement_lookup", /*tools_error*/ None).await?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "[mcp_servers.connected]\nurl = \"{replacement_url}/mcp\"\n\
             [mcp_servers.disabled]\nurl = \"{mcp_server_url}/mcp\"\nenabled = false\n"
        ))
        .write(codex_home.path())?;
    let response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                thread_id: Some(thread.id),
            },
        })
        .await?;
    assert_eq!(
        response
            .data
            .into_iter()
            .map(|status| (
                status.name,
                status.runtime_status,
                status.tools.into_keys().collect::<BTreeSet<_>>()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "connected".to_string(),
                None,
                BTreeSet::from(["replacement_lookup".to_string()])
            ),
            (
                "disabled".to_string(),
                Some(McpServerConnectionStatus::Disabled),
                BTreeSet::new()
            ),
        ]
    );
    replacement_handle.abort();
    let _ = replacement_handle.await;
    mcp_server_handle.abort();
    let _ = mcp_server_handle.await;
    Ok(())
}

#[tokio::test]
async fn mcp_server_status_list_reports_disconnected_stdio_transport() -> Result<()> {
    skip_if_remote!(
        Ok(()),
        "the stdio executable and shutdown marker are host-local"
    );
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let codex_home = TempDir::new()?;
    let exit_file = codex_home.path().join("exit-mcp-server");
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "[mcp_servers.stdio]\ncommand = {}\nstartup_timeout_sec = 2\n\
             [mcp_servers.stdio.env]\nMCP_TEST_EXIT_FILE = {}\n",
            toml::Value::String(stdio_server_bin()?),
            toml::Value::String(exit_file.to_string_lossy().into_owned()),
        ))
        .write(codex_home.path())?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let ThreadStartResponse { thread, .. } = mcp.start_thread(ThreadStartParams::default()).await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_matching_notification("stdio MCP server ready", |notification| {
            notification.method == "mcpServer/startupStatus/updated"
                && notification
                    .params
                    .as_ref()
                    .is_some_and(|params| params["name"] == "stdio" && params["status"] == "ready")
        }),
    )
    .await??;
    for expected in [
        McpServerConnectionStatus::Connected,
        McpServerConnectionStatus::Failed,
    ] {
        if expected == McpServerConnectionStatus::Failed {
            std::fs::write(&exit_file, "exit")?;
        }
        timeout(DEFAULT_READ_TIMEOUT, async {
            loop {
                let response: ListMcpServerStatusResponse = mcp
                    .request(|request_id| ClientRequest::McpServerStatusList {
                        request_id,
                        params: ListMcpServerStatusParams {
                            server_name: None,
                            cursor: None,
                            limit: None,
                            detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
                            thread_id: Some(thread.id.clone()),
                        },
                    })
                    .await?;
                let statuses = response
                    .data
                    .into_iter()
                    .map(|status| (status.name, status.runtime_status))
                    .collect::<Vec<_>>();
                if statuses == vec![("stdio".to_string(), Some(expected))] {
                    return Ok::<(), anyhow::Error>(());
                }
                sleep(Duration::from_millis(/*millis*/ 20)).await;
            }
        })
        .await??;
    }
    Ok(())
}

#[tokio::test]
async fn mcp_server_status_retains_capabilities_when_tool_discovery_fails() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (url, handle) = start_mcp_server(
        "lookup",
        Some(rmcp::ErrorData::internal_error(
            "tool discovery unavailable",
            /*data*/ None,
        )),
    )
    .await?;
    let codex_home = TempDir::new()?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!("[mcp_servers.degraded]\nurl = \"{url}/mcp\""))
        .write(codex_home.path())?;
    let mut app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    for detail in [None, Some(McpServerStatusDetail::ToolsAndAuthOnly)] {
        let response: ListMcpServerStatusResponse = app
            .request(|request_id| ClientRequest::McpServerStatusList {
                request_id,
                params: ListMcpServerStatusParams {
                    server_name: None,
                    cursor: None,
                    limit: None,
                    detail,
                    thread_id: None,
                },
            })
            .await?;
        let status = response
            .data
            .iter()
            .find(|status| status.name == "degraded")
            .expect("configured server is included");
        assert!(status.tools.is_empty());
        assert!(
            status
                .tools_error
                .as_ref()
                .is_some_and(|error| error.contains("tool discovery unavailable"))
        );
        assert_eq!(
            status
                .server_capabilities
                .as_ref()
                .and_then(|caps| caps.get("extensions")),
            Some(&json!({
                "openai/settings": { "readTool": "settings.read", "updateTool": "settings.update" }
            }))
        );
    }
    handle.abort();
    let _ = handle.await;
    Ok(())
}

#[derive(Clone)]
struct McpStatusServer {
    tool_name: Arc<String>,
    tools_error: Option<rmcp::ErrorData>,
}

impl ServerHandler for McpStatusServer {
    fn get_info(&self) -> ServerConfig {
        let mut capabilities = ServerCapabilities::builder().enable_tools().build();
        capabilities.extensions = Some(BTreeMap::from([(
            "openai/settings".to_string(),
            serde_json::from_value(
                json!({ "readTool": "settings.read", "updateTool": "settings.update" }),
            )
            .expect("settings capability is a JSON object"),
        )]));
        ServerConfig::new(capabilities).with_server_info(
            Implementation::new("lookup-server", "1.0.0").with_title("Lookup Server"),
        )
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        if let Some(error) = &self.tools_error {
            return Err(error.clone());
        }
        let input_schema: JsonObject = serde_json::from_value(json!({
            "type": "object",
            "additionalProperties": false
        }))
        .map_err(|err| rmcp::ErrorData::internal_error(err.to_string(), None))?;

        let mut tool = Tool::new(
            Cow::Owned(self.tool_name.as_ref().clone()),
            Cow::Borrowed("Look up test data."),
            Arc::new(input_schema),
        );
        tool.annotations = Some(ToolAnnotations::new().read_only(true));

        Ok(ListToolsResult::with_all_items(vec![tool]))
    }
}

#[derive(Clone)]
struct SlowInventoryServer {
    tool_name: Arc<String>,
}

impl ServerHandler for SlowInventoryServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let input_schema: JsonObject = serde_json::from_value(json!({
            "type": "object",
            "additionalProperties": false
        }))
        .map_err(|err| rmcp::ErrorData::internal_error(err.to_string(), None))?;

        let mut tool = Tool::new(
            Cow::Owned(self.tool_name.as_ref().clone()),
            Cow::Borrowed("Look up test data."),
            Arc::new(input_schema),
        );
        tool.annotations = Some(ToolAnnotations::new().read_only(true));

        Ok(ListToolsResult::with_all_items(vec![tool]))
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ListResourcesResult, rmcp::ErrorData> {
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(ListResourcesResult::with_all_items(Vec::new()))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ListResourceTemplatesResult, rmcp::ErrorData> {
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(ListResourceTemplatesResult::with_all_items(Vec::new()))
    }
}

#[tokio::test]
async fn mcp_server_status_list_tools_and_auth_only_skips_slow_inventory_calls() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (mcp_server_url, mcp_server_handle) = start_slow_inventory_mcp_server("lookup").await?;
    let codex_home = TempDir::new()?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            "[mcp_servers.some-server]\nurl = \"{mcp_server_url}/mcp\""
        ))
        .write(codex_home.path())?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;

    let request_id = mcp
        .send_list_mcp_server_status_request(ListMcpServerStatusParams {
            server_name: None,
            cursor: None,
            limit: None,
            detail: Some(McpServerStatusDetail::ToolsAndAuthOnly),
            thread_id: None,
        })
        .await?;
    let response: ListMcpServerStatusResponse =
        timeout(Duration::from_millis(500), mcp.read_response(request_id)).await??;

    assert_eq!(response.next_cursor, None);
    assert_eq!(response.data.len(), 1);
    let status = &response.data[0];
    assert_eq!(status.name, "some-server");
    assert_eq!(
        status.tools.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["lookup".to_string()])
    );
    assert_eq!(status.resources, Vec::new());
    assert_eq!(status.resource_templates, Vec::new());

    mcp_server_handle.abort();
    let _ = mcp_server_handle.await;

    Ok(())
}

#[tokio::test]
async fn mcp_server_status_list_keeps_tools_for_sanitized_name_collisions() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(Vec::new()).await;
    let (dash_server_url, dash_server_handle) =
        start_mcp_server("dash_lookup", /*tools_error*/ None).await?;
    let (underscore_server_url, underscore_server_handle) =
        start_mcp_server("underscore_lookup", /*tools_error*/ None).await?;
    let codex_home = TempDir::new()?;
    mock_responses_config(&server.uri())
        .with_extra_config(&format!(
            r#"[mcp_servers.some-server]
url = "{dash_server_url}/mcp"

[mcp_servers.some_server]
url = "{underscore_server_url}/mcp"
"#
        ))
        .write(codex_home.path())?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;
    let response: ListMcpServerStatusResponse = mcp
        .request(|request_id| ClientRequest::McpServerStatusList {
            request_id,
            params: ListMcpServerStatusParams {
                server_name: None,
                cursor: None,
                limit: None,
                detail: None,
                thread_id: None,
            },
        })
        .await?;

    assert_eq!(response.next_cursor, None);
    assert_eq!(response.data.len(), 2);
    let status_tools = response
        .data
        .iter()
        .map(|status| {
            (
                status.name.as_str(),
                status.tools.keys().cloned().collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        status_tools,
        BTreeMap::from([
            ("some-server", BTreeSet::from(["dash_lookup".to_string()])),
            (
                "some_server",
                BTreeSet::from(["underscore_lookup".to_string()])
            )
        ])
    );

    dash_server_handle.abort();
    let _ = dash_server_handle.await;
    underscore_server_handle.abort();
    let _ = underscore_server_handle.await;

    Ok(())
}

async fn start_mcp_server(
    tool_name: &str,
    tools_error: Option<rmcp::ErrorData>,
) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let tool_name = Arc::new(tool_name.to_string());
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(McpStatusServer {
                tool_name: Arc::clone(&tool_name),
                tools_error: tools_error.clone(),
            })
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let router = Router::new().nest_service("/mcp", mcp_service);

    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    Ok((format!("http://{addr}"), handle))
}

async fn start_slow_inventory_mcp_server(tool_name: &str) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let tool_name = Arc::new(tool_name.to_string());
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(SlowInventoryServer {
                tool_name: Arc::clone(&tool_name),
            })
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let router = Router::new().nest_service("/mcp", mcp_service);

    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    Ok((format!("http://{addr}"), handle))
}

fn mock_responses_config(server_uri: &str) -> MockResponsesConfig {
    MockResponsesConfig::new(server_uri)
        .with_root_config("compact_prompt = \"compact\"\nmodel_auto_compact_token_limit = 1024")
        .with_provider_config("supports_websockets = false")
}
