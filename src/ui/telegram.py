"""Telegram bot gateway for OnIt.

Allows remote interaction with an OnIt agent via a Telegram bot.
Usage: onit --gateway  (requires TELEGRAM_BOT_TOKEN env var)
"""

import asyncio
import logging
import os

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

    def __init__(self, onit, token: str):
        self.onit = onit
        self.token = token

    async def _start_command(self, update: Update, context) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Hello! I'm an OnIt agent. Send me a message and I'll help you out."
        )

    async def _handle_message(self, update: Update, context) -> None:
        """Handle incoming text messages."""
        text = update.message.text
        if not text:
            return

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
            response = await self.onit.process_task(text)
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

        # Send response, splitting if needed
        for chunk in _split_message(response):
            await update.message.reply_text(chunk)

    async def _handle_photo(self, update: Update, context) -> None:
        """Handle incoming photo messages (for vision models)."""
        caption = update.message.caption or "Describe this image."

        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        # Download the highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_path = os.path.join(self.onit.data_path, f"telegram_{photo.file_unique_id}.jpg")
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
            response = await self.onit.process_task(caption, images=[image_path])
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

        for chunk in _split_message(response):
            await update.message.reply_text(chunk)

    def run_sync(self) -> None:
        """Start the Telegram bot with its own event loop via run_polling().

        This must be called instead of `run()` because python-telegram-bot's
        run_polling() manages its own asyncio event loop.  OnIt should call
        this from a dedicated thread or before entering its own asyncio.run().
        """
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
