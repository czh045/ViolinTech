import unittest

from paper_experiments import detect_boundaries


class BoundaryDetectionTest(unittest.TestCase):
    def test_detect_boundaries_merges_close_transitions(self):
        labels = ["A", "A", "B", "B", "B", "C", "C", "A", "A"]
        boundaries = detect_boundaries(labels, min_gap=2)
        self.assertEqual(boundaries, [2, 5, 7])


if __name__ == "__main__":
    unittest.main()
