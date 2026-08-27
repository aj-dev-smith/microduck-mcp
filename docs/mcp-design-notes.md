# MCP design notes

Why this server is shaped the way it is, based on a review of the MCP
landscape as of **August 2026** (spec revision 2026-07-28, Python SDK v2).

## Tools, not resources — including for reads

MCP's conceptual split is *resources = application-controlled context,
tools = model-controlled actions*. Robot state and camera frames might sound
like "context," but they are **agent-initiated reads in a poll-act loop**, and
in practice tools are the only primitive every client supports:

- The Claude API MCP connector supports **only tools**.
- Claude Desktop exposes resources as manual UI attachments; the model won't
  read them autonomously.
- Claude Code lets the model read resources, but through generic wrapper
  tools that strip the per-tool guidance we put in descriptions.
- Resource **subscriptions** (the natural fit for live state) are supported by
  essentially one client (VS Code), and the 2026-07-28 spec revision replaced
  the whole mechanism (`resources/subscribe` → `subscriptions/listen`).

State is also stale-on-read and parameterized (`view`, `distance` on the
camera) — tool semantics, not document semantics. Published robotics MCP
servers (ros-mcp-server, mujoco-mcp) converge on the same pattern.

A `duck://state` resource may be added *in addition* someday for UI
attachment; it will never be the primary path.

## Conventions this server follows

- **Typed structured output**: tools return a `DuckState` Pydantic model, so
  the SDK publishes an `outputSchema`, validates results, and every field
  carries units and coordinate-frame docs. Floats are rounded server-side —
  17-digit telemetry is token waste for a polling client.
- **Tool annotations + titles** on every tool (`read_only_hint` on
  state/camera, `destructive_hint` on reset, `idempotent_hint` on sticky
  intents). Client enforcement is still thin in 2026, but they cost nothing
  and permission-gating on them is growing.
- **Errors the model should see are tool errors**: an unreachable simulator
  raises `ToolError` (→ `isError: true` with an actionable message), never a
  JSON-RPC protocol error the model can't recover from. This distinction
  sharpened in SDK v2, where raising `MCPError` becomes a protocol error.
- **Latency-aware tool shape**: the sim runs in real time and an LLM turn
  takes seconds, so commands are *sticky intents*, every mutating tool
  returns post-action state (sampled after a short settle) to save a poll,
  descriptions state the staleness contract explicitly, and
  `duck_drive(duration_s=...)` collapses drive/poll/stop into one call.
- **Inline images** from `duck_camera` (`Image` helper), capped at 640×480 —
  well-supported in Claude clients, unlike `resource_link` blocks.
- **stdio transport** as the shipped default for a locally spawned server.
  If remote access is ever needed: stateless Streamable HTTP via `mcp.run()`,
  never SSE (deprecated since 2025-03-26).

## Deliberately not used

- **Resources as primary interface / resource subscriptions** — see above.
- **Elicitation** (e.g. confirming reset) — form-mode unsupported in Claude
  clients; `destructive_hint` + client-side approval is the gate.
- **Sampling, Roots, MCP-level Logging** — all formally deprecated in the
  2026-07-28 revision. Server logs go to stderr.
- **SSE transport** — deprecated.

## Dependency policy

`mcp>=2,<3`. SDK v2 targets spec 2026-07-28 while transparently serving
2025-era clients, so the stateless protocol, `server/discover`, and
deterministic tool ordering are inherited without code changes. The v1 import
path (`mcp.server.fastmcp`) is deleted in v2 — code against
`mcp.server.mcpserver` only.

## Sources

- Spec changelogs: [2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog),
  [2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/changelog);
  [release blog](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- [Server concepts](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts),
  [architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)
- Python SDK v2: [what's new](https://py.sdk.modelcontextprotocol.io/whats-new/),
  [tools](https://py.sdk.modelcontextprotocol.io/servers/tools/),
  [structured output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/),
  [migration](https://py.sdk.modelcontextprotocol.io/migration/)
- Anthropic: [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents),
  [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Client support matrices: [canimcp.dev](https://canimcp.dev/),
  [apify/mcp-client-capabilities](https://github.com/apify/mcp-client-capabilities),
  [Claude Code MCP docs](https://code.claude.com/docs/en/mcp),
  [MCP connector docs](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)
- Ecosystem discussion: [Resources: the overlooked primitive](https://layered.dev/mcp-resources-the-overlooked-primitive/),
  [MCP resources underused](https://usewire.io/blog/mcp-resources-underused-half-of-mcp/)
- Robotics precedents: [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server),
  [mujoco-mcp](https://github.com/robotlearning123/mujoco-mcp)
