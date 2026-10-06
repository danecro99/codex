//! Foreground MCP 2026 tasks: bounded requests, no task-lifetime deadline.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use rmcp::model::CallToolRequestParams;
use rmcp::model::CallToolResponse;
use rmcp::model::CallToolResult;
use rmcp::model::CancelTaskParams;
use rmcp::model::GetExtensions;
use rmcp::model::GetMeta;
use rmcp::model::GetTaskParams;
use rmcp::model::InputRequest;
use rmcp::model::InputRequests;
use rmcp::model::InputResponses;
use rmcp::model::NumberOrString;
use rmcp::model::ServerRequest;
use rmcp::model::Task;
use rmcp::model::TaskPayload;
use rmcp::model::UpdateTaskParams;
use rmcp::service::RequestContext;
use rmcp::service::RoleClient;
use rmcp::service::RunningService;
use rmcp::service::Service;
use tokio::time;
use tracing::error;

use crate::elicitation_client_service::ElicitationClientService;
use crate::rmcp_client::ElicitationPauseState;
use crate::rmcp_client::active_time_timeout;

type ClientService = RunningService<RoleClient, ElicitationClientService>;
const DEFAULT_TASK_POLL_INTERVAL: Duration = Duration::from_secs(1);
// The optional ordinary-call deadline never means unbounded task protocol I/O.
const DEFAULT_TASK_IO_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_TASK_INPUT_BYTES: usize = 1024 * 1024;

// Drop is the native tool-future cancellation boundary. Never recreate a
// connection or resubmit tools/call when cancelling its server-owned task.
struct TaskCancellation {
    service: Arc<ClientService>,
    task: Option<Task>,
    io_timeout: Duration,
}

impl Drop for TaskCancellation {
    fn drop(&mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        let service = Arc::clone(&self.service);
        let io_timeout = self.io_timeout;
        tokio::spawn(async move {
            let cancelled = async {
                // A reused task ID must not turn interruption into cancellation
                // of a different task generation on the same peer.
                let before = service
                    .peer()
                    .get_task(GetTaskParams::new(task.task_id.clone()))
                    .await?;
                validate_task_identity(&task, &before.task.task)?;
                if before.task.status().is_terminal() {
                    return Ok(());
                }
                service
                    .peer()
                    .cancel_task(CancelTaskParams::new(task.task_id.clone()))
                    .await?;
                let result = service
                    .peer()
                    .get_task(GetTaskParams::new(task.task_id.clone()))
                    .await?;
                validate_task_identity(&task, &result.task.task)?;
                if !result.task.status().is_terminal() {
                    bail!("MCP task cancellation was acknowledged but not terminal");
                }
                Ok::<_, anyhow::Error>(())
            };
            match time::timeout(io_timeout, cancelled).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    error!(task_id = %task.task_id, %error, "MCP task cancellation unproven")
                }
                Err(error) => {
                    error!(task_id = %task.task_id, %error, "MCP task cancellation I/O timed out")
                }
            }
        });
    }
}

async fn request<T>(
    method: &str,
    timeout: Option<Duration>,
    pause: &ElicitationPauseState,
    operation: impl Future<Output = Result<T, rmcp::service::ServiceError>>,
) -> Result<T> {
    let Some(timeout) = timeout else {
        return operation.await.map_err(Into::into);
    };
    active_time_timeout(timeout, pause.subscribe(), operation)
        .await
        .map_err(|()| anyhow::anyhow!("MCP {method} I/O timed out after {timeout:?}"))?
        .map_err(Into::into)
}

// Dispatch through the existing native handler, including elicitation and its
// human-wait pause. MRTR and task input use identical request identities.
async fn input_responses(
    service: &ClientService,
    requests: InputRequests,
) -> Result<InputResponses> {
    let mut responses = InputResponses::new();
    for (id, input) in requests {
        let mut request = match input {
            InputRequest::CreateMessage(request) => ServerRequest::CreateMessageRequest(request),
            InputRequest::Elicitation(request) => ServerRequest::ElicitRequest(request),
            InputRequest::ListRoots(request) => ServerRequest::ListRootsRequest(request),
            _ => bail!("unsupported MCP input request"),
        };
        let mut context = RequestContext::new(
            NumberOrString::String(Arc::from(id.as_str())),
            service.peer().clone(),
        );
        context.meta = std::mem::take(request.get_meta_mut());
        context.extensions = std::mem::take(request.extensions_mut());
        let result = service.service().handle_request(request, context).await?;
        responses.insert(id, serde_json::to_value(result)?);
    }
    Ok(responses)
}

fn validate_task_identity(expected: &Task, received: &Task) -> Result<()> {
    if received.task_id != expected.task_id || received.created_at != expected.created_at {
        bail!("MCP task identity changed on the bound connection");
    }
    Ok(())
}

pub(crate) async fn call_tool(
    service: Arc<ClientService>,
    params: CallToolRequestParams,
    timeout: Option<Duration>,
    pause: ElicitationPauseState,
) -> Result<CallToolResult> {
    let response = request(
        "tools/call",
        timeout,
        &pause,
        service.call_tool_once(params.clone()),
    )
    .await?;
    continue_tool_call(service, params, response, timeout, pause).await
}

pub(crate) async fn continue_tool_call(
    service: Arc<ClientService>,
    mut params: CallToolRequestParams,
    mut response: CallToolResponse,
    timeout: Option<Duration>,
    pause: ElicitationPauseState,
) -> Result<CallToolResult> {
    loop {
        match response {
            CallToolResponse::Complete(result) => return Ok(result),
            CallToolResponse::InputRequired(result) => {
                let responses =
                    input_responses(&service, result.input_requests.unwrap_or_default()).await?;
                if responses.is_empty() && result.request_state.is_none() {
                    bail!("MCP input_required has neither inputs nor request state");
                }
                if responses.is_empty() {
                    time::sleep(Duration::from_millis(50)).await;
                }
                params.input_responses = (!responses.is_empty()).then_some(responses);
                params.request_state = result.request_state;
                response = request(
                    "tools/call",
                    timeout,
                    &pause,
                    service.call_tool_once(params.clone()),
                )
                .await?;
            }
            CallToolResponse::Task(created) => {
                let io_timeout = match timeout {
                    Some(duration) => duration,
                    None => DEFAULT_TASK_IO_TIMEOUT,
                };
                let timeout = Some(io_timeout);
                if created.task.task_id.is_empty() || created.task.created_at.is_empty() {
                    bail!("MCP task has an empty identity");
                }
                let mut cancellation = TaskCancellation {
                    service: Arc::clone(&service),
                    task: Some(created.task.clone()),
                    io_timeout,
                };
                let mut handled_inputs = BTreeMap::<String, serde_json::Value>::new();
                let mut handled_input_bytes = 0usize;
                loop {
                    let result = request(
                        "tasks/get",
                        timeout,
                        &pause,
                        service
                            .peer()
                            .get_task(GetTaskParams::new(created.task.task_id.clone())),
                    )
                    .await?;
                    validate_task_identity(&created.task, &result.task.task)?;
                    match result.task.payload {
                        TaskPayload::Completed { result } => {
                            cancellation.task = None;
                            return serde_json::from_value(serde_json::Value::Object(result))
                                .context("MCP task completed with an invalid tool result");
                        }
                        TaskPayload::Failed { error } => {
                            cancellation.task = None;
                            bail!("MCP task failed: {}", serde_json::Value::Object(error));
                        }
                        TaskPayload::Cancelled => {
                            cancellation.task = None;
                            bail!("MCP task was cancelled");
                        }
                        TaskPayload::InputRequired { input_requests } => {
                            if input_requests.is_empty() {
                                bail!("MCP task input_required has no input requests");
                            }
                            let mut new_requests = InputRequests::new();
                            for (id, input) in input_requests {
                                let encoded = serde_json::to_value(&input)?;
                                if let Some(previous) = handled_inputs.get(&id) {
                                    if previous != &encoded {
                                        bail!("MCP task input request identity changed");
                                    }
                                    continue;
                                }
                                handled_input_bytes +=
                                    id.len() + serde_json::to_vec(&encoded)?.len();
                                if handled_input_bytes > MAX_TASK_INPUT_BYTES {
                                    bail!(
                                        "MCP task input identity storage limit exceeded: {MAX_TASK_INPUT_BYTES} bytes"
                                    );
                                }
                                handled_inputs.insert(id.clone(), encoded);
                                new_requests.insert(id, input);
                            }
                            // tasks/update acknowledgements are eventually
                            // consistent. Never elicit or submit an input twice.
                            if !new_requests.is_empty() {
                                let responses = input_responses(&service, new_requests).await?;
                                request(
                                    "tasks/update",
                                    timeout,
                                    &pause,
                                    service.peer().update_task(UpdateTaskParams::new(
                                        created.task.task_id.clone(),
                                        responses,
                                    )),
                                )
                                .await?;
                            }
                        }
                        TaskPayload::Working => {}
                        _ => bail!("unsupported MCP task payload"),
                    }
                    let interval = match result.task.task.poll_interval_ms {
                        Some(0) => bail!("MCP task pollIntervalMs must be positive"),
                        Some(millis) => Duration::from_millis(millis),
                        None => DEFAULT_TASK_POLL_INTERVAL,
                    };
                    time::sleep(interval).await;
                    if service.is_closed() {
                        bail!("MCP task connection closed");
                    }
                }
            }
            _ => bail!("unsupported MCP tools/call response"),
        }
    }
}
