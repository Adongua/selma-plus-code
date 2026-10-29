"""CPU-only label alignment and Tiny-ImageNet annotation checks."""

import importlib.util
from collections import Counter
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

    def test_validation_class_folders_agree_with_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "val" / "n00000001"
            folder.mkdir(parents=True)
            image = folder / "val_0.JPEG"
            image.touch()
            annotations = root / "val" / "val_annotations.txt"
            annotations.write_text("val_0.JPEG\tn00000001\n")
            self.assertEqual(validator.validation_items(root, self.classes), [(image, 1)])
            annotations.write_text("val_0.JPEG\tn00000002\n")
            with self.assertRaisesRegex(ValueError, "class directory disagrees"):
                validator.validation_items(root, self.classes)
            annotations.write_text("missing.JPEG\tn00000001\n")
            with self.assertRaisesRegex(ValueError, "1 annotated validation images are missing"):
                validator.validation_items(root, self.classes)

    def test_first_name_mapping_records_wordnet_alias(self):
        labels = dict(self.labels)
        original = labels[0]
        labels[0] = "n99999999"
        names = {wnid: f"class {i}" for i, wnid in enumerate(self.classes)}
        names[labels[0]] = names[original]
        classes, indices, rows = validator.class_mapping_by_first_name(self.classes, labels, names)
        self.assertEqual(classes, self.classes)
        self.assertEqual(indices, list(reversed(range(200))))
        self.assertEqual(sum(row["different_wnid"] for row in rows), 1)
        self.assertEqual(rows[-1]["teacher_wnid"], "n99999999")
        with self.assertRaises(ValueError):
            validator.class_mapping(self.classes, labels)

    def test_first_name_rejects_ambiguity_and_missing_ids(self):
        names = {wnid: f"class {i}" for i, wnid in enumerate(self.classes)}
        names[self.classes[1]] = names[self.classes[0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            validator.class_mapping_by_first_name(self.classes, self.labels, names)
        del names[self.classes[0]]
        with self.assertRaisesRegex(ValueError, "missing WordNet IDs"):
            validator.class_mapping_by_first_name(self.classes, self.labels, names)

    def test_summary_includes_empty_and_partial_groups(self):
        support = Counter({2: 4, 4: 2})
        top1 = Counter({2: 3, 4: 1})
        top5 = Counter({2: 4, 4: 2})
        summary = validator.accuracy_summary([2, 4, 9], support, top1, top5)
        self.assertEqual(summary["samples"], 6)
        self.assertEqual(summary["classes_observed"], 2)
        self.assertEqual(summary["top1"], 4 / 6)
        self.assertEqual(summary["top5"], 1.0)
        self.assertIsNone(validator.accuracy_summary([9], support, top1, top5)["top1"])


if __name__ == "__main__":
    unittest.main()
