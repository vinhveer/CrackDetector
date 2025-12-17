import unittest

import cv2
import numpy as np

from src.postprocess.geometry_filter_v3 import GeometryRulesConfig, hard_reject_region


def _make_blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


class TestGeometryFilterV3(unittest.TestCase):
    def test_keep_thin_line(self):
        mask = _make_blank()
        mask[32, 10:54] = 1  # thin horizontal crack
        cfg = GeometryRulesConfig()
        dropped, reason, feats = hard_reject_region(mask, cfg)
        self.assertFalse(dropped)
        self.assertIsNone(reason)
        self.assertGreater(feats["skeleton_length"], 0)

    def test_reject_blob_like(self):
        mask = _make_blank()
        mask[20:44, 20:44] = 1  # solid square blob
        cfg = GeometryRulesConfig()
        dropped, reason, _ = hard_reject_region(mask, cfg)
        self.assertTrue(dropped)
        self.assertEqual(reason, "blob_like")

    def test_reject_closed_loop(self):
        mask = _make_blank()
        cv2.circle(mask, (32, 32), 14, 1, thickness=8)  # thick ring loop
        cfg = GeometryRulesConfig()
        dropped, reason, feats = hard_reject_region(mask, cfg)
        self.assertGreater(feats["skeleton_length"], 0)
        self.assertTrue(dropped)
        self.assertEqual(reason, "closed_loop")

    def test_reject_too_thick(self):
        mask = _make_blank()
        mask[26:38, 8:56] = 1  # thick straight line
        cfg = GeometryRulesConfig(t_width_mean_high=10.0, t_width_var_low=2.0)
        dropped, reason, feats = hard_reject_region(mask, cfg)
        self.assertGreater(feats["skeleton_length"], 0)
        self.assertTrue(dropped)
        self.assertEqual(reason, "too_thick")

    def test_reject_ornament_like(self):
        mask = _make_blank()
        cv2.line(mask, (8, 8), (56, 8), 1, thickness=7)
        cv2.line(mask, (56, 8), (56, 56), 1, thickness=7)
        cfg = GeometryRulesConfig(t_curv_var_low=0.2, t_width_var_low=5.0)
        dropped, reason, feats = hard_reject_region(mask, cfg)
        self.assertGreater(feats["skeleton_length"], 0)
        self.assertTrue(dropped)
        self.assertEqual(reason, "ornament_like")


if __name__ == "__main__":
    unittest.main()
