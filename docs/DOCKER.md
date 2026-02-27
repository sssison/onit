# Docker

## Build the image

```bash
docker build -t onit .
```

## Run the container

**Interactive terminal mode:**

```bash
docker run -it --rm --env-file .env onit
```

**Web UI (Gradio on port 9000):**

```bash
docker run -it --rm -p 9000:9000 --env-file .env onit --web --web-port 9000
```

**A2A server:**

```bash
docker run -it --rm -p 9001:9001 --env-file .env onit --a2a --a2a-port 9001
```

**Telegram gateway:**

```bash
docker run -it --rm --env-file .env onit --gateway telegram
```

Requires `TELEGRAM_BOT_TOKEN` in your `.env` file.

**Viber gateway:**

```bash
docker run -it --rm -p 8443:8443 --env-file .env onit --gateway viber
```

Requires `VIBER_BOT_TOKEN` and `VIBER_WEBHOOK_URL` in your `.env` file. Port 8443 must be exposed for the webhook server.

**With a custom config:**

```bash
docker run -it --rm -v $(pwd)/configs:/app/configs --env-file .env onit --config configs/default.yaml
```

## Docker Compose

Start the web UI and A2A server together:

```bash
docker compose up --build
```

This launches services defined in `docker-compose.yml`:

| Service | Description | Port |
|---------|-------------|------|
| `onit-mcp` | MCP servers | 18200-18204 |
| `onit-web` | Web UI | 9000 |
| `onit-a2a` | A2A server | 9001 |
| `onit-gateway` | Telegram bot gateway | — |
| `onit-viber` | Viber bot gateway | 8443 |

The `onit-web` and `onit-a2a` services depend on `onit-mcp` and connect to it via the `--mcp-host` flag.

## Session persistence

Each service uses a named Docker volume to persist per-user session data (chat history JSONL files) across container restarts:

- `web-sessions` — Web UI sessions
- `a2a-sessions` — A2A server sessions
- `gateway-sessions` — Telegram/Viber gateway sessions

Each browser tab (web), A2A context, or messaging chat gets its own isolated session with separate chat history and file storage.

## Environment variables

Pass API keys via an `.env` file or individual `-e KEY=value` flags. Never bake secrets into the image.

Example `.env` file:

```bash
ONIT_HOST=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OLLAMA_API_KEY=your_ollama_key
OPENWEATHER_API_KEY=your_weather_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # required for --gateway telegram
VIBER_BOT_TOKEN=your_viber_bot_token        # required for --gateway viber
VIBER_WEBHOOK_URL=https://your-domain.com/viber  # public HTTPS URL for Viber webhook
```
