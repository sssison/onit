# Release Notes

## v0.1.4

### New Features

- **Per-Session Isolation (Web UI)** — Each browser tab now gets its own independent session with isolated chat history, file storage, and response routing. Multiple users can chat concurrently without seeing each other's messages or files. Sessions auto-cleanup after 24 hours.
- **Per-Session Isolation (A2A Server)** — Each A2A context (client conversation) gets its own isolated session with separate chat history, data directory, and safety queue. Different A2A clients no longer share state.
- **Concurrent Request Processing (Web UI)** — Web UI requests are now processed concurrently via `process_task()` (matching the Telegram/Viber gateway pattern), instead of sequentially through a single queue.

### Improvements

- **Session-Scoped File Routes** — File uploads and downloads are now scoped per session (`/uploads/{session_id}/{filename}`), preventing file conflicts between users. Legacy `/uploads/{filename}` route preserved for backward compatibility.
- **Per-Session Stop** — The Stop button in the web UI now only cancels the current browser tab's task, not all users' tasks. A2A client disconnects similarly only cancel that client's in-flight request.

## v0.1.3a

### New Features

- **Viber Gateway** — Chat with OnIt remotely via a Viber bot. Supports text and photo messages with vision processing. Requires a public HTTPS webhook URL (see [Gateway Quick Start](docs/GATEWAY_QUICK_START.md)).
- **Gateway Auto-Detection** — `onit --gateway` now auto-detects Telegram or Viber based on which environment variable is set (`TELEGRAM_BOT_TOKEN` or `VIBER_BOT_TOKEN`).
- **Tunnel Documentation** — Comprehensive guide for tunneling options: Cloudflare Tunnel, ngrok, localtunnel, Tailscale Funnel, and SSH reverse tunnel.

### Improvements

- **Friendly Error Messages** — All user-facing interfaces (terminal, web, Telegram, Viber, A2A) now return friendly messages instead of exposing internal error details. Server errors are logged via `logger.error()` for debugging.
- **Webhook Registration Timing** — Fixed race condition where Viber webhook was registered before uvicorn started accepting connections.
- **Logging** — Added `logger` to `chat.py` and `onit.py` so API errors (timeouts, connection failures) are always logged regardless of `verbose` setting.

### Bug Fixes

- Fixed `chat()` returning raw error strings to users on `APITimeoutError` and `OpenAIError` — now returns `None` so callers handle it consistently.
- Fixed `agent_session()` sending raw exception text to the output queue — now sends `None` to trigger the retry prompt.
- Fixed Telegram and Viber gateways exposing `f"Error: {e}"` to users on exceptions.

## v0.1.2

### New Features

- **Telegram Gateway** — Chat with OnIt remotely via a Telegram bot. Supports text and photo messages with vision processing.
- **VLM Integration** — Send images to A2A servers for vision-language model processing (`--a2a-image` flag).
- **Remote MCP Servers** — Connect to external MCP servers using `--mcp-sse` and `--mcp-host` flags.
- **Unified Tools MCP Server** — Consolidated web search, bash, filesystem, and document tools into a single `ToolsMCPServer`.

### Improvements

- **Standalone App Refactor** — OnIt is now a fully self-contained package installable via `pip install onit==0.1.2`.
- **Simplified CLI** — Streamlined command-line options and argument parsing.
- **Docker Compose** — Multi-service orchestration with `onit-mcp`, `onit-web`, `onit-a2a`, and `onit-gateway` services.
- **Prompt Engineering** — Date-aware prompts and improved prompt template handling.
- **Error Handling** — Better error recovery and user-facing error messages across all interfaces.

### Bug Fixes

- Fixed vLLM kwargs handling for tool calls.
- Fixed message formatting across terminal, web, and Telegram UIs.
- Resolved test failures and improved test coverage.
- Security fixes for bash and filesystem MCP servers.

### Dependencies

- Added: `a2a-sdk[all]`, `beautifulsoup4`, `python-dateutil`, `geopy`, `ddgs`, `pypdf`, `ollama`, `urllib3`, `PyMuPDF`
- Optional: `python-telegram-bot` (gateway), `google-auth` (web)
- Removed: `requirements.txt` (dependencies now managed entirely via `pyproject.toml`)
- Removed: Google Workspace and Microsoft Office MCP servers (moved to separate packages)

## v0.1.1

- Initial public release with MCP tool integration, web UI, and A2A protocol support.
