"""Regression checks for slow local image generation."""

import tempfile
import unittest
from unittest.mock import patch

import ollama_image_agent as agent


class ImageWorkflowTests(unittest.TestCase):
    def test_render_finishing_after_nineteen_minutes_is_returned(self):
        prompt_id = "slow-render"
        clock = [0]

        def history(_path):
            if clock[0] < 19 * 60:
                return {}
            return {prompt_id: {"status": {"status_str": "success"}, "outputs": {
                "9": {"images": [{"filename": "picnic.png", "subfolder": "", "type": "output"}]}
            }}}

        with tempfile.TemporaryDirectory() as directory, patch.object(agent, "OUTPUT_DIR", directory), \
             patch.object(agent, "_comfy_post", return_value={"prompt_id": prompt_id}), \
             patch.object(agent, "_comfy_get", side_effect=history), \
             patch.object(agent.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(agent.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
             patch.object(agent.urllib.request, "urlretrieve") as retrieve:
            result = agent._run_workflow({})

        self.assertTrue(result.endswith("picnic.png"))
        self.assertGreaterEqual(clock[0], 19 * 60)
        retrieve.assert_called_once()

    def test_engine_error_is_reported_without_waiting_an_hour(self):
        with patch.object(agent, "_comfy_post", return_value={"prompt_id": "failed"}), \
             patch.object(agent, "_comfy_get", return_value={"failed": {
                 "status": {"status_str": "error"}, "outputs": {}
             }}):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                agent._run_workflow({})


if __name__ == "__main__":
    unittest.main()
