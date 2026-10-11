# Prompt: Split PixelForge into crates, and a complete MCP server

Run this **after** `PROMPT_PSD.md` and `PROMPT_PARITY.md`. By then:
- PSD / PSB is the native format;
- documents can be 8, 16 or 32-bit, with HDR and raw;
- feature parity with Photoshop is closed (except AI);
- there is a typed **command bus** (every undoable operation is a command with parameters, used
  by Actions, Batch and the scripting API).

You are working in `~/Desktop/Code/Rust/pixelforge`. It is Rust with GPUI via `gpui-kit` 0.7.1 /
`gpui-pre` 0.3.8, vendored and patched in `vendor/`. It is not a git repo, so back up the tree to
the scratchpad before you start, and again before each phase.

Your job:

1. **Split the single `pixelforge` crate into a Cargo workspace** of focused crates with clean
   boundaries, no behaviour change, and faster builds.
2. **Build a complete MCP server** (Model Context Protocol) so any MCP client (Claude Code,
   Claude Desktop, IDEs, other agents) can do everything a user can do in PixelForge. That
   means every command, document and layer query, rendering, presets, automation and
   long-running jobs. The server runs both headless and inside the running app, where the user
   watches and can stop the agent. Security and undo safety come first.

## What the research says (read this first; confirm on the current pages)

**The MCP specification.** The current revision is **2026-07-28**; the previous one is
2025-11-25. Read the full spec and its changelog before designing. The key changes in
2026-07-28:

- **Stateless:**
  - No `initialize` / `notifications/initialized` handshake and no protocol sessions
    (`Mcp-Session-Id` is gone). Every request carries its protocol version and client
    capabilities in `_meta`.
  - Servers **must** implement `server/discover` (versions, capabilities, identity).
  - Cross-call state uses explicit, server-minted handles passed as tool arguments.
- **`subscriptions/listen`:** one long-lived stream for opted-in change notifications (tool,
  prompt and resource list changes, resource subscriptions). It replaces `resources/subscribe`
  and the HTTP GET endpoint. Progress and log notifications flow on the response stream of
  their own request.
- **Multi Round-Trip Requests (MRTR):** instead of server-initiated requests (sampling,
  elicitation, roots), a server returns `resultType: "input_required"` with `inputRequests`.
  The client retries the original request with `inputResponses`. Servers keep their own
  correlation in `requestState`. All results carry `resultType`.
- **Tasks:** long-running requests became an official extension (`io.modelcontextprotocol/tasks`),
  polled with `tasks/get` and fed with `tasks/update`.
- **Caching:** list and read results carry `ttlMs` and `cacheScope`, and servers *should*
  return `tools/list` in a deterministic order (prompt-cache friendly).
- **Schemas:** `inputSchema` and `outputSchema` accept any JSON Schema 2020-12 keywords, with
  `$ref` resolution rules; `structuredContent` can be any JSON value.
- **Headers:** Streamable HTTP POSTs carry `Mcp-Method` and `Mcp-Name`; tool parameters can map
  to custom headers (`x-mcp-header`).
- **Tracing:** OpenTelemetry trace context travels in `_meta` (`traceparent`, `tracestate`,
  `baggage`).
- **Deprecated:** Roots, Sampling and Logging (new implementations should not adopt them: take
  directories from configuration or tool parameters, log to stderr or OpenTelemetry), the old
  HTTP+SSE transport, and Dynamic Client Registration (in favour of Client ID Metadata
  Documents).
- **Error codes:** resource-not-found is now -32602. Codes -32020 to -32099 are reserved for the
  spec.

Support 2026-07-28 first. Also stay compatible with 2025-11-25 clients (the `initialize`
handshake, sessions, server-initiated elicitation), since many hosts will lag. Negotiate by
version and test both.

**Extensions worth supporting:**
- **MCP Apps** (`io.modelcontextprotocol/ui`, SEP-1865, final): tools can declare `ui://`
  resources (`text/html;profile=mcp-app`) that hosts render in a sandboxed iframe, talking to the
  host over MCP's JSON-RPC. Its spec lives in the `ext-apps` repository; read its current
  revision.
- **Tasks**, as above.

**Rust SDK.** `rmcp` is the official Rust SDK (github.com/modelcontextprotocol/rust-sdk,
Apache-2.0). The latest seen is 3.5.1, with features for the server, macros, stdio
(`transport-io`), Streamable HTTP server, auth, elicitation and `request-state`. Check which
protocol revisions it implements. If it lags behind 2026-07-28, wrap or extend it rather than
forking, and say so. It needs my approval as a dependency, along with its tree (tokio,
schemars…).

**Tool design for LLM clients.**
- Clients put every tool's name, description and schema into the model's context, and many
  clients cap or degrade with hundreds of tools.
- A complete server therefore needs **progressive discovery**: a small, stable core toolset plus
  a way to find and call the rest. Keep descriptions short, consistent and example-led.
- Image results cost a lot of context; return scaled previews and resource links rather than
  full-resolution pixels.

## Ground rules

- **Dependencies:** ask me before any new crate or download (name, source, version, size,
  licence), including `rmcp` and its tree, any HTTP server stack, JSON Schema tooling, an
  OpenTelemetry exporter, and the MCP Inspector or any Node tooling for testing. Prefer what's
  already in `Cargo.lock`.
- **No behaviour change from the split:**
  - Every test, benchmark budget, `PF_DEMO` screenshot, PSD corpus result and Photoshop
    comparison stays identical (timings within noise; measure).
  - Build warning-free, `cargo test --release` green with `PF_GPU=off` and `force`.
- **The command bus is the single source of truth:**
  - The UI, Actions, Batch, scripting and MCP all go through it.
  - MCP never gets a private code path that the UI can't reach, or the other way round.
  - Every MCP mutation is a normal undoable history step, labelled with the client's name.
- **Proof bar:** never claim an MCP feature works without:
  - an automated test through a real MCP client (an `rmcp` client in the test suite) over both
    stdio and Streamable HTTP;
  - a `PF_DEMO=tools` screenshot of the running app showing an agent's change and the agent
    activity indicator.
  Look at every screenshot and fix what's wrong.

## Phase 1: Split into a workspace

1. **Measure first:** clean and incremental build times (touching one engine file, one UI file,
   one GPU file), binary size, test time, and the full benchmark set. These are the baseline.
2. **Design the crate graph** in `docs/ARCHITECTURE.md` before moving code. Use acyclic,
   one-way dependencies. A starting point to refine against the real module graph (check the
   imports, don't trust this list):

   | Crate | Contents |
   |---|---|
   | `pf-core` | pixels, `px` SWAR math, tiles and grids, mips, colour, metrics, the memory budget, scratch disk |
   | `pf-doc` | document model, layers, masks, selections, annotations, history |
   | `pf-compose` | compositing, blend modes, layer styles |
   | `pf-vector` | paths, path ops, shapes, text, hyphenation, warp |
   | `pf-filters` | filters, Filter Gallery, catalog |
   | `pf-retouch` | brushes and paint, retouch, Poisson, PatchMatch, Liquify, Perspective Warp, Vanishing Point |
   | `pf-select` | segmentation, graph cut, live-wire, refine, objects |
   | `pf-gpu` | the wgpu runtime and kernels, behind a feature on the crates that use it |
   | `pf-psd` | the PSD / PSB codec and descriptor codec |
   | `pf-io` | other formats, raw, HDR files, export |
   | `pf-color` | colour management |
   | `pf-ml` | the optional ML backend, kept off by default |
   | `pf-commands` | the typed command bus, command schemas (JSON Schema), Actions and Batch |
   | `pf-script` | the scripting engine |
   | `pf-engine` | a thin facade that headless users depend on |
   | `pf-app` | the GPUI front end |
   | `pf-mcp` | the MCP server |
   | `pf-cli` | the `pixelforge` binary: app launch, `--bench*`, `--compare-gaps`, `--psd-corpus`, `--batch`, `mcp` subcommands |
   | `pf-bench` / `pf-testkit` | benchmark harnesses, test photos and helpers, Photoshop drivers |

   Keep `vendor/` as is. Put shared dependency versions in `[workspace.dependencies]`, lints in
   `[workspace.lints]`, and profiles at the workspace root.
3. **Move incrementally, one crate per step.**
   - Build, test and run benchmarks after each move. Keep a log of each step with timings.
   - Make crate APIs explicit (`pub` only what's used across crates; `pub(crate)` elsewhere).
   - Remove the cycles you find by moving types down or adding traits, not with feature-flag
     tricks.
   - Tests move with their code.
4. **Feature flags:** `gpu` (default on), `ml`, `mcp`, `script`, `gui`. A headless build
   (`pf-engine` + `pf-mcp` + `pf-cli` without `gui`) must compile without GPUI and run on a
   machine without a display.
5. **Report** the before / after table (clean and incremental builds, binary size, test time,
   benchmarks). Incremental builds after an engine change should be much faster.

## Phase 2: The command model for MCP

1. **Schemas:** every command in `pf-commands` gets:
   - a stable name (namespaced: `layer.duplicate`, `filter.gaussian_blur`,
     `adjust.curves`…), a title, a one-line description and a usage example;
   - JSON Schema 2020-12 for its parameters, with units, ranges, enums with titles, and
     defaults;
   - an output schema;
   - its annotations: read-only, destructive, idempotent, open-world, long-running, needs a
     selection, needs a layer kind, supported bit depths and colour modes.
   Generate the schemas from the Rust types (ask before adding a schema crate). A test checks
   every command's schema against its parameter struct by round trip.
2. **Handles:** documents, layers, channels, paths, selections, history states, presets and
   tasks get stable, server-minted IDs that survive renames and reorders (as the stateless spec
   requires).
3. **Query surface:** read-only commands for everything the UI shows:
   - the document tree;
   - layer properties and bounds;
   - pixel samples and region statistics (histogram, mean / median, colour clusters);
   - selection bounds and area;
   - text contents and styles;
   - history;
   - presets, fonts and preferences;
   - measurements;
   - the parity matrix.
4. **Rendering:** render the composite, a layer, a mask, a channel or a region at a requested
   size and format (PNG / WebP / JPEG), optionally with a before / after split. Cached and
   cheap for repeated previews.

## Phase 3: The MCP server (`pf-mcp`)

1. **Transports:**
   - **stdio** via `pixelforge mcp` (headless) and Streamable HTTP via
     `pixelforge mcp --http [addr]`, bound to `127.0.0.1` by default.
   - For HTTP: a per-install bearer token, origin checking (DNS-rebinding protection), and the
     `Mcp-Method` / `Mcp-Name` header checks.
   - Remote exposure is off unless configured, and then requires the spec's OAuth 2.1
     authorization (Client ID Metadata Documents, `iss` validation).
2. **Embedded in the app:**
   - The running PixelForge exposes the same server (Preferences ▸ Automation ▸ "Allow MCP
     connections", off by default), so agents work on the user's open documents live.
   - Show agent activity in the UI: a status bar indicator, the current tool call, a pause and
     stop button, and history steps labelled "via <client>".
   - Commands from MCP and from the user are serialised through the command bus. The user's
     input always wins, and a long-running agent job never blocks the UI.
3. **Protocol:**
   - 2026-07-28 (stateless, `server/discover`, `resultType`, MRTR, `subscriptions/listen`,
     `CacheableResult`, deterministic list order, `_meta` version and capability handling,
     OpenTelemetry trace context).
   - 2025-11-25 compatibility (`initialize`, sessions, `resources/subscribe`, server-initiated
     elicitation).
   - Version negotiation tested both ways.
4. **Tools:**
   - **A core toolset**, small and stable, that covers common work directly: open / create /
     save / export a document; get the document tree; render a preview; select (by colour
     range, subject, rectangle, path, layer transparency); add and edit layers; apply a filter
     or adjustment by name with parameters; transform; text; undo / redo / history; run an
     action; batch.
   - **Every command** from the bus is reachable through progressive discovery:
     - `commands.search` (by text, category, layer kind);
     - `commands.describe` (full schema and examples);
     - `commands.run` (any command by name with validated parameters).
     Also expose the most used commands as first-class tools, chosen by measurement (see
     Phase 5). Keep the total listed tools within what major clients handle well; document the
     number and why.
   - **Results:** structured results (`structuredContent` with the output schema, plus a short
     text summary), image content as scaled previews, and resource links for anything large.
   - **Errors:** actionable messages (what failed, why, how to fix: "needs a pixel layer;
     layer 'Title' is text, rasterize it first or pick another layer"), never stack traces.
5. **Resources**, with templates, read caching and subscriptions:
   - `pixelforge://documents`, `…/doc/{id}` (manifest), `…/doc/{id}/layers`,
     `…/doc/{id}/layer/{lid}` (properties) and `…/layer/{lid}/thumbnail{?size}`;
   - `…/doc/{id}/composite{?size,format}`, `…/selection`, `…/channels`, `…/history`;
   - `pixelforge://presets/{kind}`, `…/fonts`, `…/actions`, `…/preferences`;
   - `pixelforge://docs/{topic}` (the user guide, generated from the README and the command
     schemas).
   Change notifications fire when the user or another agent edits.
6. **Prompts**, as workflow templates with arguments and argument completion (layer names,
   presets, fonts, actions): portrait retouch, cut out a subject with a shadow, prepare a print
   file, social media sizes, batch-export a folder, colour-grade to a reference, focus-stack, a
   panorama. Each one guides the model through the core tools.
7. **Completion:** `completion/complete` for every argument that has a finite or searchable
   domain (layer names, blend modes, fonts, presets, action names, file paths inside the
   allowed directories).
8. **Input requests (MRTR, and elicitation for older clients):**
   - confirm destructive operations (flatten, delete a layer, overwrite a file, discard
     unsaved changes), unless the client passed explicit confirmation;
   - ask for missing required parameters with a form schema.
   Never request secrets.
9. **Tasks extension:** batch runs, large filters, Content-Aware Fill, raw development, PSD
   corpus runs and exports run as tasks, with progress, partial results, cancellation (which
   undoes cleanly) and `tasks/update` for mid-task input.
10. **MCP Apps** (`io.modelcontextprotocol/ui`), for hosts that support them: `ui://` HTML
    resources for:
    - a live document viewer (zoom, pan, before / after slider);
    - a layers panel;
    - a curves and levels editor;
    - a colour picker;
    - a batch-results gallery.
    They talk back through tool calls, so every interaction is still a command. Sandbox-safe:
    no external resources, no network, CSP set.
11. **Observability:** structured logs to stderr and the Info panel, OpenTelemetry trace
    propagation from `_meta` (ask before any exporter crate), and per-tool latency statistics.

## Phase 4: Security and safety

1. **File access:**
   - Only within allowed directories, set in Preferences, by `--allow-dir` flags, or in the
     server configuration (not Roots, which is deprecated).
   - Canonicalise every path, deny traversal and symlink escape, and never overwrite without a
     confirmation or an explicit `overwrite: true`.
2. **Modes:**
   - read-only (queries and renders only);
   - edit (no file writes outside the allowed directories);
   - full.
   Set per transport and per client. Each tool's annotations are accurate (a test checks every
   destructive command is marked destructive).
3. **Untrusted content:** text layer contents, file metadata, layer names and filenames are
   returned as data, clearly delimited, and never interpreted as instructions by the server.
   Document this for clients.
4. **Limits:** request size, image result size, concurrent tasks, and rate limits per client,
   all with clear errors.
5. **Undo safety:** every agent change is undoable from the History panel. "Revert agent
   session" undoes everything a client did since it connected.
6. **Audit log:** every call (client, tool, arguments, result summary, duration) goes to a
   local log, viewable in the app.

## Phase 5: Quality — make it great to use from an LLM

1. **Evaluation suite** (`pf-mcp/evals/`): 50 or more realistic tasks in natural language.
   Examples:
   - "remove the background and put the subject on white";
   - "make the sky warmer";
   - "batch-resize this folder to 2048 px and export WebP";
   - "fix the red eye";
   - "add a title in a bold sans at the top".
   Each has a programmatic check on the result (pixel metrics, the layer tree, files written).
2. **Run the suite with a real model through an MCP client.** Driving Claude Code headless
   against the server is the natural choice; ask me before using any API key or paid service.
   Measure:
   - success rate;
   - tool calls per task;
   - tokens spent on tool listings and results;
   - wall time.
3. **Iterate on the design from those numbers:** tool names, descriptions, the core toolset,
   result shapes, preview sizes, error messages. Report the before / after numbers per change.
4. **Conformance:**
   - Test against the MCP Inspector and the official conformance tests, if the spec project
     publishes them (ask before installing Node tooling).
   - Test against at least two real hosts: Claude Code (`claude mcp add`), and Claude Desktop
     if available.
   - Record the results.

## Phase 6: Documentation

- **README "MCP server" section:**
  - what it can do, transports, and how to connect (config snippets for Claude Code, Claude
    Desktop and generic stdio and HTTP clients);
  - security modes and allowed directories;
  - the core toolset, progressive discovery, resources, prompts, tasks and MCP Apps;
  - the evaluation results.
- **A generated tool and command catalogue** (`docs/mcp-tools.md`) from the schemas, kept in
  sync by a test.
- **`docs/ARCHITECTURE.md`**: the crate graph, what goes where, and the rules (dependency
  direction, the command bus as the only mutation path).
- **README layout section:** rewritten for the workspace.

## Verification per phase

1. **Phase 1:** identical test results, benchmarks within noise, the build-time table, and a
   headless build without GPUI.
2. **Tests:** schema round trips for every command; the handle stability tests; the MCP protocol
   tests over stdio and HTTP for both protocol revisions (discover, list, call, MRTR, tasks,
   subscriptions, completion, cancellation, errors); the security tests (path escapes,
   read-only mode, confirmations, limits); undo of agent sessions.
3. **App:** an agent driving the open app while the user works, with screenshots of the
   activity indicator, history labels, pause and stop. Look at every screenshot.
4. **Evals:** the suite run, with numbers.
5. **README and docs** updated as above.

## Reporting

After each phase, report:
- what was built;
- build-time and benchmark numbers (Phase 1);
- protocol and conformance results, and eval numbers;
- which screenshots you checked and what they showed;
- anything deferred or blocked, with the reason (for example, an `rmcp` gap or a host lacking
  MCP Apps support).

## Sources

- MCP specification, revision 2026-07-28: https://modelcontextprotocol.io/specification/2026-07-28
- MCP changelog since 2025-11-25:
  https://modelcontextprotocol.io/specification/2026-07-28/changelog
- MCP 2025-11-25 changelog (tasks, elicitation changes):
  https://modelcontextprotocol.io/specification/2025-11-25/changelog
- SEP-1865, MCP Apps:
  https://modelcontextprotocol.io/seps/1865-mcp-apps-interactive-user-interfaces-for-mcp
- Official Rust SDK (`rmcp`, Apache-2.0): https://github.com/modelcontextprotocol/rust-sdk ·
  https://crates.io/crates/rmcp
- Cargo workspaces: https://doc.rust-lang.org/cargo/reference/workspaces.html
