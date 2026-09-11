# Live MCP catalog refresh: custom implementation contract

This custom 0.153.4 change preserves the existing private auth-home, runtime
MCP, durable history and compaction contracts. It backports OpenAI commit
`32351a7b1ae3d6106fb2b7adccfbd6051e375ce6`'s client-owned catalog foundation;
it is not an upgrade to the complete 0.154 release.

## Ownership and update boundary

The live `RmcpClient` records a monotonically increasing revision for
`notifications/tools/list_changed`. Reading this revision never consumes a
notification. `ClientToolCatalog` owns that client's catalog, served notification
revision, publication revision, and refresh/call synchronization. A reused
client retains the same catalog; a new client is a different binding identity.

Before `McpRuntime` captures the next model binding, it re-lists each notified
ready client through that same MCP connection. Only that client's fetches are
serialized. Other servers do not share its refresh lock. Names, descriptions,
schemas and policy metadata come from the complete paginated `tools/list` result,
then pass through the existing tool filters and normalization.

A notification received during fetching remains pending. A binding racing that
notification reports the affected server unavailable until a subsequent model
boundary lists it; it never silently publishes the preceding catalog as current.
Process-cache invalidation also fences fetches started before invalidation, so
an older in-flight response cannot resurrect the invalidated snapshot.

## Calls and failure visibility

An immutable `PreparedMcpCall` keeps its captured client, configuration and
catalog revision. Before irreversible preparation it checks both publication
and notification revisions. A stale call is refused before preparation.
Calls already admitted may finish; publishing a replacement waits for their
read guards. Neither refreshing nor failure handling replays any tool call.

A failed re-list is stored as an error, not `Ok([])` or the preceding tools.
The binding exposes `catalog_errors` with `needs_mcp_tool_catalog_refresh` and
the server's diagnostic. Core appends changed availability to conversation
history through `InternalModelContextFragment` and emits a warning, including
recovery. The diagnostic display uses the existing 8,000-byte truncation policy;
the stored error remains intact. Other servers' valid tools remain available.
An unchanged failed notification revision is not retried automatically. Another
notification or an explicitly replaced connection provides a new observation.
A successful empty catalog is valid and carries no refresh error.

## Scope and adoption

This changes native MCP discovery/binding, not provider identity, permissions,
tool approval or server deployment. Central META MCP Proxy generations still
own upstream code and catalog publication. Its adapter additionally requires
the same stdio session to list the offered form before ordinary tool calls.

An old running Codex executable cannot acquire this implementation from disk.
The initial binary/adapter adoption therefore remains a separately coordinated
runtime cutover. Later catalog changes use MCP notifications and do not require
another Codex restart. Direct CapaBench process ownership is a separate routing
change; this implementation does not silently move it into a personal catalog.

## Regression boundaries

- `binding_tests.rs`: notification before preparation, notification during fetch,
  failed versus empty catalog, no repeated protocol-error retry, stale calls,
  and existing active-call publication guards.
- `connection_manager_tests.rs` and `runtime.rs`: client reuse, binding cache
  invalidation, exact-client Apps inventory and shared-cache races.
- `tool_catalog_cache.rs`: a pre-notification in-flight fetch cannot republish.
- `core/tests/suite/mcp_tool_cache.rs`: actual next model request after a real
  stdio MCP notification, added/removed tools, same-name schema changes, same
  MCP process, failed re-list diagnostic and successful empty catalog.
