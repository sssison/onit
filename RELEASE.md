# Release Notes

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
