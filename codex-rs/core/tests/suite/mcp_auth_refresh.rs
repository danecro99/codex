#![allow(clippy::unwrap_used)]

use anyhow::Result;
use codex_config::McpServerTransportConfig;
use codex_core::config::ConfigBuilder;
use codex_core::config::Constrained;
use codex_core::plugins_manager_for_config;
use codex_exec_server_test_support::environment_manager_without_environments;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::ExternalAuth;
use codex_login::ExternalAuthFuture;
use codex_login::ExternalAuthRefreshContext;
use codex_mcp::CODEX_APPS_MCP_SERVER_NAME;
use codex_mcp::CodexAppsToolsCache;
use codex_mcp::EffectiveMcpServer;
use codex_mcp::McpRuntime;
use codex_mcp::McpRuntimeContext;
use codex_mcp::McpRuntimeInput;
use codex_mcp::McpStartupPolicy;
use codex_mcp::McpToolCatalogCache;
use codex_protocol::mcp::ClientMcpExtensions;
use codex_protocol::protocol::AskForApproval;
use core_test_support::apps_test_server::AppsTestServer;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

// Installs a known snapshot through AuthManager's public external-auth path.
struct StaticExternalAuth(CodexAuth);

impl ExternalAuth for StaticExternalAuth {
    fn resolve(&self) -> ExternalAuthFuture<'_, CodexAuth> {
        Box::pin(async { Ok(self.0.clone()) })
    }

    fn refresh(&self, _context: ExternalAuthRefreshContext) -> ExternalAuthFuture<'_, CodexAuth> {
        Box::pin(async { Ok(self.0.clone()) })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hosted_plugin_runtime_ps_mcp_tool_calls_rebuild_for_the_current_account() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let apps_server = AppsTestServer::mount_hosted_plugin_runtime_searchable(&server).await?;
    let home = Arc::new(TempDir::new()?);
    let initial_auth = CodexAuth::from_external_chatgpt_tokens(
        "header.e30.account-a",
        "account-a",
        /*chatgpt_plan_type*/ None,
    )?;
    let auth_manager = AuthManager::from_auth_for_testing_with_home(
        initial_auth.clone(),
        home.path().to_path_buf(),
    );
    // Build the hosted-plugin config directly so the local test origin can
    // exercise the connection-manager auth path. Effective server resolution
    // correctly strips ChatGPT auth from untrusted localhost origins.
    let mut hosted_plugin_runtime_config = codex_mcp::hosted_plugin_runtime_mcp_server_config(
        &apps_server.chatgpt_base_url,
        /*apps_mcp_product_sku*/ None,
        /*originator*/ None,
    );
    let McpServerTransportConfig::StreamableHttp {
        bearer_token_env_var,
        ..
    } = &mut hosted_plugin_runtime_config.transport
    else {
        panic!("hosted plugin runtime should use streamable HTTP");
    };
    // Keep the test on the AuthManager path even if the developer has the
    // debug bearer override in their environment.
    *bearer_token_env_var = None;
    let mcp_servers = HashMap::from([(
        CODEX_APPS_MCP_SERVER_NAME.to_string(),
        EffectiveMcpServer::configured(hosted_plugin_runtime_config),
    )]);
    let mut config = ConfigBuilder::default()
        .codex_home(home.path().to_path_buf())
        .build()
        .await?;
    config.permissions.approval_policy = Constrained::allow_any(AskForApproval::Never);
    let plugins_manager = plugins_manager_for_config(&config, Arc::clone(&auth_manager));
    let mcp_config = Arc::new(config.to_mcp_config(&plugins_manager).await);
    let runtime_context = McpRuntimeContext::new(
        Arc::new(environment_manager_without_environments()),
        home.path().to_path_buf(),
    );
    let runtime_a = McpRuntime::new(McpRuntimeInput {
        startup_policy: McpStartupPolicy::Eager,
        config: Arc::clone(&mcp_config),
        plugins_available: false,
        ready_selected_capability_roots: Vec::new(),
        mcp_servers: mcp_servers.clone(),
        submit_id: "test".to_string(),
        tx_event: None,
        startup_cancellation_token: CancellationToken::new(),
        runtime_context: runtime_context.clone(),
        codex_apps_tools_cache: CodexAppsToolsCache::default(),
        tool_catalog_cache: McpToolCatalogCache::default(),
        codex_apps_tools_cache_key: codex_mcp::codex_apps_tools_cache_key(Some(&initial_auth)),
        client_mcp_extensions: ClientMcpExtensions::default(),
        auth: Some(initial_auth.clone()),
        auth_manager: Some(Arc::clone(&auth_manager)),
        elicitation_reviewer: None,
        elicitation_lifecycle: None,
    })
    .await;
    // The model-provider test covers AuthManager reload behavior. Keep this
    // regression focused on core MCP wiring by updating the same shared
    // manager after the MCP client has been created.
    let auth_b = CodexAuth::from_external_chatgpt_tokens(
        "header.e30.account-b",
        "account-b",
        /*chatgpt_plan_type*/ None,
    )?;
    auth_manager
        .set_external_auth(Arc::new(StaticExternalAuth(auth_b.clone())))
        .await?;

    // An A-bound runtime must not borrow B from the shared AuthManager.
    let tool_result = runtime_a
        .latest_call_tool(
            CODEX_APPS_MCP_SERVER_NAME,
            "calendar_create_event",
            /*environment_id*/ None,
            Some(json!({
                "title": "Lunch",
                "starts_at": "2026-06-18T12:00:00Z",
            })),
            /*meta*/ None,
            /*requested_timeout*/ None,
            /*wait_for_server*/ true,
        )
        .await?;
    assert_eq!(tool_result.is_error, Some(false));

    // Rebuilding from the current session account must use B, without carrying
    // A's client or credentials into the replacement runtime.
    let runtime_b = McpRuntime::new(McpRuntimeInput {
        startup_policy: McpStartupPolicy::Eager,
        config: mcp_config,
        plugins_available: false,
        ready_selected_capability_roots: Vec::new(),
        mcp_servers,
        submit_id: "test".to_string(),
        tx_event: None,
        startup_cancellation_token: CancellationToken::new(),
        runtime_context,
        codex_apps_tools_cache: CodexAppsToolsCache::default(),
        tool_catalog_cache: McpToolCatalogCache::default(),
        codex_apps_tools_cache_key: codex_mcp::codex_apps_tools_cache_key(Some(&auth_b)),
        client_mcp_extensions: ClientMcpExtensions::default(),
        auth: Some(auth_b),
        auth_manager: Some(Arc::clone(&auth_manager)),
        elicitation_reviewer: None,
        elicitation_lifecycle: None,
    })
    .await;
    let tool_result = runtime_b
        .latest_call_tool(
            CODEX_APPS_MCP_SERVER_NAME,
            "calendar_create_event",
            /*environment_id*/ None,
            Some(json!({
                "title": "Lunch",
                "starts_at": "2026-06-18T12:00:00Z",
            })),
            /*meta*/ None,
            /*requested_timeout*/ None,
            /*wait_for_server*/ true,
        )
        .await?;
    assert_eq!(tool_result.is_error, Some(false));

    let requests = server
        .received_requests()
        .await
        .expect("mock server should capture tool-call requests");
    let tool_call_requests = requests
        .iter()
        .filter(|request| {
            request.url.path() == "/api/codex/ps/mcp"
                && serde_json::from_slice::<Value>(&request.body)
                    .ok()
                    .is_some_and(|body| {
                        body.get("method").and_then(Value::as_str) == Some("tools/call")
                    })
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_call_requests.len(), 2);

    let stale_account_request = tool_call_requests[0];
    assert_eq!(
        stale_account_request
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        None
    );
    assert_eq!(
        stale_account_request
            .headers
            .get("chatgpt-account-id")
            .and_then(|value| value.to_str().ok()),
        None
    );

    let current_account_request = tool_call_requests[1];
    assert_eq!(
        current_account_request
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer header.e30.account-b")
    );
    assert_eq!(
        current_account_request
            .headers
            .get("chatgpt-account-id")
            .and_then(|value| value.to_str().ok()),
        Some("account-b")
    );

    Ok(())
}
