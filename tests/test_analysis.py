import unittest

import numpy as np
import pandas as pd

from analysis import (
    condition_tests,
    infer_column_mapping,
    participant_features,
    prepare_data,
    spatial_metrics,
)


class AnalysisTests(unittest.TestCase):
    def test_infers_both_repository_schemas(self):
        viewgauss = infer_column_mapping(
            ["Frame", "PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "RotW"]
        )
        hololens = infer_column_mapping(
            ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )
        self.assertEqual(viewgauss["time"], "Frame")
        self.assertEqual(viewgauss["qw"], "RotW")
        self.assertEqual(hololens["time"], "timestamp")
        self.assertEqual(hololens["qx"], "qx")

    def test_dotnet_ticks_and_quaternion_sign_are_handled(self):
        ticks = 637139238387771648
        raw = pd.DataFrame(
            {
                "timestamp": [ticks, ticks + 100_000, ticks + 200_000],
                "x": [0.0, 0.1, 0.2],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
                "qx": [0.0, 0.0, 0.0],
                "qy": [0.0, 0.0, 0.0],
                "qz": [0.0, 0.0, 0.0],
                "qw": [2.0, -2.0, 2.0],
            }
        )
        mapping = infer_column_mapping(raw.columns)
        data, report = prepare_data(raw, mapping, smoothing_window=1)
        self.assertEqual(report.timestamp_mode, "dotnet_ticks")
        np.testing.assert_allclose(data["elapsed_s"], [0.0, 0.01, 0.02], atol=1e-8)
        np.testing.assert_allclose(data["speed"].dropna(), [10.0, 10.0])
        np.testing.assert_allclose(data["angular_speed"].dropna(), [0.0, 0.0])
        self.assertEqual(report.quaternion_repairs, 3)

    def test_frame_rate_and_coordinate_transform(self):
        raw = pd.DataFrame(
            {
                "Frame": [1, 2, 3],
                "PosX": [0.0, 0.0, 0.0],
                "PosY": [0.0, 0.0, 0.0],
                "PosZ": [0.0, 1.0, 2.0],
                "RotX": [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)],
                "RotY": [0.0, 0.0, 0.0],
                "RotZ": [0.0, 0.0, 0.0],
                "RotW": [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)],
            }
        )
        data, _ = prepare_data(
            raw,
            infer_column_mapping(raw.columns),
            frame_rate=50,
            coordinate_convention="unity_to_right_handed",
            smoothing_window=1,
        )
        np.testing.assert_allclose(data["elapsed_s"], [0.0, 0.02, 0.04])
        np.testing.assert_allclose(data["z"], [0.0, -1.0, -2.0])
        # Quaternion signs are globally ambiguous; qx * qw < 0 identifies the
        # expected -90° rotation about X after the handedness conversion.
        np.testing.assert_allclose(data["qx"] * data["qw"], [-0.5] * 3)

    def test_spatial_entropy_and_independent_sample_guard(self):
        points = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
        entropy, coverage, occupied, total = spatial_metrics(points, bins=2)
        self.assertAlmostEqual(entropy, 1.0)
        self.assertEqual(occupied, 2)
        self.assertEqual(total, 8)
        self.assertAlmostEqual(coverage, 0.25)

        raw = pd.DataFrame(
            {
                "time": [0, 1, 2], "x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0],
                "qx": [0, 0, 0], "qy": [0, 0, 0], "qz": [0, 0, 0], "qw": [1, 1, 1],
            }
        )
        prepared, _ = prepare_data(raw, infer_column_mapping(raw.columns), timestamp_mode="seconds")
        features = participant_features(prepared)
        self.assertTrue(condition_tests(features).empty)


if __name__ == "__main__":
    unittest.main()
