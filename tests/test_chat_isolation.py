"""Regression checks for replies that finish after switching chats."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import qwen_image_app as app_module


class QueuedRoot:
    def __init__(self):
        self.callbacks = []

    def after(self, _delay, callback, *args):
        self.callbacks.append((callback, args))

    def drain(self):
        while self.callbacks:
            callback, args = self.callbacks.pop(0)
            callback(*args)


class ChatIsolationTests(unittest.TestCase):
    def test_late_reply_stays_in_original_chat(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app_module, "HISTORY_DIR", directory):
            app = object.__new__(app_module.ChatApp)
            app.root = QueuedRoot()
            app.session_id = "picnic"
            app.session_title = "Picnic"
            picnic_messages = [{"role": "user", "content": "make a picnic scene"}]
            cat_messages = [{"role": "user", "content": "make a cat picture"}]
            app.messages = picnic_messages
            app._active_turns = {"cat": {"id": "cat", "title": "Cat", "model": "test-model", "messages": cat_messages},
                                 "picnic": {"id": "picnic", "title": "Picnic", "model": "test-model", "messages": picnic_messages}}
            app._discarded_sessions = set()
            app.refresh_history_list = Mock()
            app.render_messages = Mock()
            app.send_btn = Mock()

            with patch.object(app_module.ollama, "chat", return_value={"message": {"role": "assistant", "content": "Here is your cat"}}):
                app.run_turn(app._active_turns["cat"])
            app.root.drain()

            self.assertIs(app.messages, picnic_messages)
            self.assertEqual(len(picnic_messages), 1)
            self.assertEqual(cat_messages[-1]["content"], "Here is your cat")
            saved = json.loads((Path(directory) / "cat.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["messages"][-1]["content"], "Here is your cat")
            app.render_messages.assert_not_called()
            app.send_btn.config.assert_not_called()

    def test_late_image_tool_result_stays_in_original_chat(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app_module, "HISTORY_DIR", directory):
            app = object.__new__(app_module.ChatApp)
            app.root = QueuedRoot()
            app.session_id = "picnic"
            app.session_title = "Picnic"
            app.messages = [{"role": "user", "content": "make a picnic scene"}]
            cat_messages = [{"role": "user", "content": "make a cat picture"}]
            cat = {"id": "cat", "title": "Cat", "model": "test-model", "messages": cat_messages}
            app._active_turns = {"cat": cat}
            app._discarded_sessions = set()
            app.refresh_history_list = Mock()
            app.render_messages = Mock()
            app.send_btn = Mock()
            app.append = Mock()
            calls = [
                {"message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "generate_image", "arguments": {"prompt": "a cat"}}}]}},
                {"message": {"role": "assistant", "content": ""}},
                {"message": {"role": "assistant", "content": "Your cat image is ready"}},
            ]
            with patch.object(app_module.ollama, "chat", side_effect=calls), patch.dict(
                app_module.AVAILABLE_FUNCTIONS, {"generate_image": Mock(return_value="cat.png")}
            ):
                app.run_turn(cat)
            app.root.drain()

            self.assertEqual(len(app.messages), 1)
            self.assertEqual(cat_messages[-1]["content"], "Your cat image is ready")
            self.assertEqual(cat_messages[-2]["content"], "cat.png")
            app.append.assert_not_called()
            app.render_messages.assert_not_called()
            saved = json.loads((Path(directory) / "cat.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["messages"][-1]["content"], "Your cat image is ready")


if __name__ == "__main__":
    unittest.main()
