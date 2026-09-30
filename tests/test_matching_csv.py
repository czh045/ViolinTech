import tempfile
import unittest
from pathlib import Path

from violin_feature_utils import find_matching_csv


class FindMatchingCsvTests(unittest.TestCase):
    def test_finds_source_csv_for_prepared_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "boxian" / "01"
            source_dir.mkdir(parents=True)
            (source_dir / "right01-1_mediapipe .csv").write_text("frame;value\n0;1\n", encoding="utf-8")
            prepared_wav = root / "prepared" / "boxian" / "01" / "right01-1.wav"
            prepared_wav.parent.mkdir(parents=True)
            prepared_wav.write_bytes(b"fake wav")

            csv_path = find_matching_csv(prepared_wav, base_dir=root)
            self.assertIsNotNone(csv_path)
            self.assertEqual(csv_path, source_dir / "right01-1_mediapipe .csv")


if __name__ == "__main__":
    unittest.main()
