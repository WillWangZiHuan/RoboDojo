"""CPU-only protocol tests; these do not claim to validate Isaac physics."""

from copy import deepcopy
import unittest

from experiments.seven_blocks.protocol import COLORS, Episode, make_layout, validate_control


class SevenBlocksTests(unittest.TestCase):
    def test_scene_clones_one_asset_without_mutating_source(self):
        template = {"Table": {"scale": [1.4, 1.1, 0.05]}, "Rigid": {"block": [
            {"category_idx": 5, "scale": [1, 1, 1], "default_pos": [0, 0, 0.8],
             "default_ori": [1, 0, 0, 0], "prim_path": "/old", "physics": {}}]},
                    "Clutter": {"unwanted": []}}
        original = deepcopy(template)
        layout = make_layout(template)
        blocks = layout["Rigid"]["block"]
        self.assertEqual(template, original)
        self.assertEqual(len(blocks), 7)
        self.assertEqual({b["label"] for b in blocks}, {f"block_{c}" for c in COLORS})
        self.assertEqual(len({tuple(b["default_pos"]) for b in blocks}), 7)
        self.assertTrue(all(b["category_idx"] == 5 and b["scale"] == [1, 1, 1] for b in blocks))
        self.assertNotIn("Clutter", layout)
        self.assertTrue(all("prim_path" not in b for b in blocks))
        blocks[0]["scale"][0] = 3
        self.assertEqual(blocks[1]["scale"], [1, 1, 1])

    def test_retries_preserve_history_but_do_not_complete(self):
        episode = Episode()
        for _ in range(20):
            episode.record_action(10)
            episode.reset_scene("try again")
        self.assertEqual(episode.status, "running")
        self.assertEqual((episode.actions, episode.physics_steps, episode.resets), (20, 200, 20))

    def test_submit_is_explicit_terminal_and_immutable(self):
        episode = Episode()
        with self.assertRaises(ValueError):
            episode.submit("", "reason", {})
        self.assertEqual(episode.status, "running")
        state = {"blocks": [1]}
        result = episode.submit("my work", "satisfied", state)
        state["blocks"].append(2)
        result["snapshot"]["blocks"].append(3)
        self.assertEqual(episode.submission["snapshot"], {"blocks": [1]})
        for operation in (lambda: episode.record_action(1), lambda: episode.reset_scene("retry"),
                          lambda: episode.submit("new", "new", {}), lambda: episode.interrupt("stop")):
            with self.assertRaises(RuntimeError):
                operation()

    def test_interruption_is_never_submission(self):
        episode = Episode()
        episode.interrupt("scheduler walltime")
        self.assertEqual(episode.status, "interrupted")
        self.assertIsNone(episode.submission)

    def test_invalid_commands_are_rejected(self):
        expected = {"arm": {"position": [0, 0]}, "gripper": {"position": [0, 0]}}
        validate_control(expected, expected)
        for command in ({}, {"arm": {"position": [0, 0]}},
                        {**expected, "arm": {"position": [0, float("nan")]}},
                        {**expected, "arm": {"position": [0]}},
                        {**expected, "arm": {"position": [0, 0], "velocity": [0]}}):
            with self.assertRaises(ValueError):
                validate_control(command, expected)


if __name__ == "__main__":
    unittest.main()
