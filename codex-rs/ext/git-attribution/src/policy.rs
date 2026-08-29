use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use codex_backend_client::Client as BackendClient;
use codex_extension_api::ExtensionData;
use codex_http_client::HttpClientFactory;
use codex_login::AuthManager;
use codex_login::RefreshTokenError;
use tokio::time::timeout;

#[derive(Clone, Debug)]
pub(super) struct GitAttributionPolicy {
    pub(super) auth_generation: u64,
    pub(super) enabled: bool,
}

pub(super) struct GitAttributionRetry {
    pub(super) auth_generation: u64,
    pub(super) retry_at: Instant,
}

pub(super) fn retry_deferred(thread_store: &ExtensionData, auth_generation: u64) -> bool {
    thread_store
        .get::<GitAttributionRetry>()
        .is_some_and(|retry| {
            retry.auth_generation == auth_generation && retry.retry_at > Instant::now()
        })
}

pub(super) fn cached_attribution_policy(
    thread_store: &ExtensionData,
    turn_store: &ExtensionData,
    auth_generation: u64,
) -> Option<GitAttributionPolicy> {
    thread_store
        .get::<GitAttributionPolicy>()
        .filter(|policy| policy.auth_generation == auth_generation)
        .or_else(|| {
            turn_store
                .get::<GitAttributionPolicy>()
                .filter(|policy| policy.auth_generation == auth_generation)
        })
        .map(|policy| policy.as_ref().clone())
}

#[cfg(not(test))]
const POLICY_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const POLICY_RESOLUTION_TIMEOUT: Duration = Duration::from_millis(500);
pub(super) const POLICY_RETRY_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug)]
pub(super) enum GitAttributionPolicyError {
    Auth(RefreshTokenError),
    Timeout(tokio::time::error::Elapsed),
}

impl std::fmt::Display for GitAttributionPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth(error) => write!(formatter, "failed to load auth: {error}"),
            Self::Timeout(error) => write!(formatter, "policy resolution timed out: {error}"),
        }
    }
}

impl std::error::Error for GitAttributionPolicyError {}

pub(super) async fn resolve_attribution_policy(
    auth_manager: &Arc<AuthManager>,
    base_url: &str,
    http_client_factory: &HttpClientFactory,
) -> Result<Option<GitAttributionPolicy>, GitAttributionPolicyError> {
    timeout(POLICY_RESOLUTION_TIMEOUT, async {
        let mut recovery_generation = auth_generation(auth_manager);
        let mut auth_recovery = auth_manager.unauthorized_recovery();
        loop {
            let auth_generation_at_start = auth_generation(auth_manager);
            if auth_generation_at_start != recovery_generation {
                auth_recovery = auth_manager.unauthorized_recovery();
                recovery_generation = auth_generation_at_start;
            }
            let auth = auth_manager
                .auth()
                .await
                .map_err(GitAttributionPolicyError::Auth)?;
            if auth_generation(auth_manager) != auth_generation_at_start {
                continue;
            }
            let enabled = match auth {
                Some(auth) if auth.uses_codex_backend() => {
                    let client =
                        BackendClient::from_auth(base_url, &auth, http_client_factory.clone());
                    let settings = client.get_user_settings().await;
                    if auth_generation(auth_manager) != auth_generation_at_start {
                        continue;
                    }
                    match settings {
                        Ok(settings) => Some(settings.commit_attribution_enabled),
                        Err(err) if err.is_unauthorized() && auth_recovery.has_next() => {
                            if auth_recovery.next().await.is_ok() {
                                recovery_generation = auth_generation(auth_manager);
                                continue;
                            }
                            None
                        }
                        Err(_) => None,
                    }
                }
                Some(_) | None => Some(false),
            };
            if auth_generation(auth_manager) != auth_generation_at_start {
                continue;
            }
            return Ok(enabled.map(|enabled| GitAttributionPolicy {
                auth_generation: auth_generation_at_start,
                enabled,
            }));
        }
    })
    .await
    .map_err(GitAttributionPolicyError::Timeout)?
}

pub(super) fn auth_generation(auth_manager: &AuthManager) -> u64 {
    *auth_manager.auth_change_receiver().borrow()
}
