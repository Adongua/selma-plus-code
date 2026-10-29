"""CPU-only label alignment and Tiny-ImageNet annotation checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "validate_teacher.py"
spec = importlib.util.spec_from_file_location("teacher_validation", path)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class TeacherValidationTests(unittest.TestCase):
    def setUp(self):
        self.classes = [f"n{index:08d}" for index in range(200)]
        self.labels = {i: value for i, value in enumerate(reversed(self.classes))}

    def test_explicit_mapping_reorders_teacher_logits(self):
        classes, permutation = validator.class_mapping(self.classes, self.labels)
        self.assertEqual(classes, self.classes)
        self.assertEqual(permutation, list(reversed(range(200))))

    def test_rejects_ambiguous_or_incomplete_labels(self):
        self.labels[0] = "goldfish"
        with self.assertRaisesRegex(ValueError, "WordNet IDs"):
            validator.class_mapping(self.classes, self.labels)
        with self.assertRaises(ValueError):
            validator.class_mapping(self.classes[:-1], self.labels)

    def test_annotations_require_existing_unique_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "val" / "images"
            images.mkdir(parents=True)
            (images / "a.JPEG").touch()
            annotations = root / "val" / "val_annotations.txt"
            annotations.write_text("a.JPEG\tn00000001\t0\t0\t0\t0\n")
            self.assertEqual(validator.validation_items(root, self.classes), [(images / "a.JPEG", 1)])
            annotations.write_text("a.JPEG\tn00000001\na.JPEG\tn00000002\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validator.validation_items(root, self.classes)


if __name__ == "__main__":
    unittest.main()
