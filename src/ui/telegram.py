"""Telegram bot gateway for OnIt.

Allows remote interaction with an OnIt agent via a Telegram bot.
Usage: onit --gateway  (requires TELEGRAM_BOT_TOKEN env var)
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
import tempfile

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit within Telegram's limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline boundary
        split_at = text.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


class TelegramGateway:
    """Telegram bot that routes messages through an OnIt agent instance."""

    def __init__(self, onit, token: str, verbose: bool = False):
        self.onit = onit
        self.token = token
        self.verbose = verbose
        # Per-chat session state: chat_id -> {session_id, session_path, data_path}
        self._chat_sessions: dict[int, dict] = {}

    def _get_chat_session(self, chat_id: int) -> dict:
        """Get or create session state for a Telegram chat."""
        if chat_id not in self._chat_sessions:
            session_id = str(uuid.uuid4())
            sessions_dir = os.path.dirname(self.onit.session_path)
            session_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if not os.path.exists(session_path):
                with open(session_path, "w", encoding="utf-8") as f:
                    f.write("")
            data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
            os.makedirs(data_path, exist_ok=True)
            self._chat_sessions[chat_id] = {
                "session_id": session_id,
                "session_path": session_path,
                "data_path": data_path,
            }
            logger.info("Created new session %s for Telegram chat %s", session_id, chat_id)
        return self._chat_sessions[chat_id]

    async def _start_command(self, update: Update, context) -> None:
        """Handle /start command."""
        name = "Assistant"
        await update.message.reply_text(
            f"Hey there! I'm {name}, your friendly AI assistant. "
            f"Feel free to send me a message or a photo and I'll be happy to help!\n\n"
            f"⚠️ May produce inaccurate information. Verify important details independently."
        )

    async def _handle_message(self, update: Update, context) -> None:
        """Handle incoming text messages."""
        text = update.message.text
        if not text:
            return

        chat_id = update.message.chat.id
        session = self._get_chat_session(chat_id)

        if self.verbose:
            user = update.message.from_user
            name = user.username or user.first_name or str(user.id)
            print(f"[MSG] {name}: {text}")

        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        # Keep sending typing action while processing
        stop_typing = asyncio.Event()

        async def typing_loop():
            while not stop_typing.is_set():
                try:
                    await update.message.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)

        typing_task = asyncio.create_task(typing_loop())

        try:
            response = await self.onit.process_task(
                text,
                session_id=session["session_id"],
                session_path=session["session_path"],
                data_path=session["data_path"],
            )
        except Exception as e:
            logger.error("Error processing task: %s", e)
            response = f"Error: {e}"
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if self.verbose:
            print(f"[BOT] {response}")

        # Send response, splitting if needed
        for chunk in _split_message(response):
            await update.message.reply_text(chunk)

    async def _handle_photo(self, update: Update, context) -> None:
        """Handle incoming photo messages (for vision models)."""
        caption = update.message.caption or "Describe this image."

        chat_id = update.message.chat.id
        session = self._get_chat_session(chat_id)

        if self.verbose:
            user = update.message.from_user
            name = user.username or user.first_name or str(user.id)
            print(f"[IMG] {name}: {caption}")

        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        # Download the highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_path = os.path.join(session["data_path"], f"telegram_{photo.file_unique_id}.jpg")
        await file.download_to_drive(image_path)

        # Keep sending typing action while processing
        stop_typing = asyncio.Event()

        async def typing_loop():
            while not stop_typing.is_set():
                try:
                    await update.message.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)

        typing_task = asyncio.create_task(typing_loop())

        try:
            response = await self.onit.process_task(
                caption,
                images=[image_path],
                session_id=session["session_id"],
                session_path=session["session_path"],
                data_path=session["data_path"],
            )
        except Exception as e:
            logger.error("Error processing image task: %s", e)
            response = f"Error: {e}"
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if self.verbose:
            print(f"[BOT] {response}")

        for chunk in _split_message(response):
            await update.message.reply_text(chunk)

    def run_sync(self) -> None:
        """Start the Telegram bot with its own event loop via run_polling().

        This must be called instead of `run()` because python-telegram-bot's
        run_polling() manages its own asyncio event loop.  OnIt should call
        this from a dedicated thread or before entering its own asyncio.run().
        """
        # Suppress noisy library logs; verbose mode uses print() for messages
        logging.getLogger("telegram").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        app = (
            Application.builder()
            .token(self.token)
            .concurrent_updates(True)
            .build()
        )

        app.add_handler(CommandHandler("start", self._start_command))
        app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        print("Telegram gateway running (Ctrl+C to stop)")
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
