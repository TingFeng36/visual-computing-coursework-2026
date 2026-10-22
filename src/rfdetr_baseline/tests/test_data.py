import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import verify_upstream
from data_tools import class_names, convert, validate_dataset, yolo_boxes
from PIL import Image


@contextmanager
def test_workspace():
    parent = Path(__file__).resolve().parents[1] / "verification" / "test_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(dir=parent)
    target = Path(temporary.name).resolve()
    if not target.is_relative_to(parent.resolve()):
        raise RuntimeError("Temporary test directory escaped the workspace")
    try:
        yield target
    finally:
        temporary.cleanup()


class DataContractTests(unittest.TestCase):
    def test_non_square_box_and_last_class(self):
        boxes, clipped = yolo_boxes("2 0.5 0.5 0.2 0.4", 200, 100, 3)
        self.assertEqual(boxes, [(3, [80.0, 30.0, 40.0, 40.0])])
        self.assertEqual(clipped, 0)

    def test_invalid_labels_are_rejected(self):
        for text in ["1.5 .5 .5 .2 .2", "3 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 0 .2", "0 .5 .5 .2 .2 0"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                yolo_boxes(text, 200, 100, 3)

    def test_clipping_is_explicit(self):
        with self.assertRaises(ValueError):
            yolo_boxes("0 0.05 0.5 0.2 0.2", 100, 100, 1)
        boxes, count = yolo_boxes("0 0.05 0.5 0.2 0.2", 100, 100, 1, allow_clip=True)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(boxes[0][1][2], 15)

    def test_sparse_names_are_rejected(self):
        with self.assertRaises(ValueError):
            class_names({0: "a", 2: "b"})

    def fixture(self, root):
        for split in ("train", "val"):
            (root/"images"/split).mkdir(parents=True)
            (root/"labels"/split).mkdir(parents=True)
            Image.new("RGB", (200,100), "red" if split == "train" else "blue").save(root/"images"/split/"one.png")
            (root/"labels"/split/"one.txt").write_text("1 .5 .5 .2 .4\n", encoding="utf-8")
        yaml = root/"data.yaml"
        yaml.write_text("train: images/train\nval: images/val\nnames: [first, last]\n", encoding="utf-8")
        return yaml

    def test_full_conversion_alias_and_tamper_detection(self):
        with test_workspace() as temp:
            root=Path(temp)
            source=self.fixture(root/"source")
            report=convert(source,root/"output",validation_as_test=True,copy_images=True)
            self.assertTrue(report["test_is_validation_alias"])
            annotations=json.loads((root/"output/train/_annotations.coco.json").read_text())
            self.assertEqual(annotations["annotations"][0]["category_id"],2)
            self.assertEqual([c["id"] for c in annotations["categories"]],[1,2])
            annotations["annotations"][0]["bbox"][0]=0
            (root/"output/train/_annotations.coco.json").write_text(json.dumps(annotations))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                validate_dataset(root/"output")

    def test_test_split_is_not_silently_created(self):
        with test_workspace() as temp:
            root=Path(temp)
            source=self.fixture(root/"source")
            with self.assertRaisesRegex(ValueError,"test folder"):
                convert(source,root/"output",copy_images=True)

    def test_train_val_overlap_is_rejected(self):
        with test_workspace() as temp:
            root=Path(temp)
            source=self.fixture(root/"source")
            source.write_text("train: images/train\nval: images/train\nnames: [first, last]\n")
            with self.assertRaisesRegex(ValueError,"overlap"):
                convert(source,root/"output",validation_as_test=True,copy_images=True)

    def test_vendor_provenance(self):
        self.assertEqual(verify_upstream()["commit"],"03aa832eb07126b8bbd0fff45790b1f2ad45be2b")


if __name__ == "__main__":
    unittest.main()
