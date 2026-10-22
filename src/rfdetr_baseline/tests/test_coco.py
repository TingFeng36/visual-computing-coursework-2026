import json
import unittest
from coco_tools import inspect_coco, synthetic, subset
from test_data import test_workspace


class NativeCocoTests(unittest.TestCase):
    def test_sparse_category_90_and_explicit_subset(self):
        with test_workspace() as temp:
            root = temp / "coco"
            audit = synthetic(root)
            self.assertEqual(audit["num_classes"],90)
            self.assertEqual(audit["actual_categories"],80)
            self.assertIn(90,audit["category_names"])
            self.assertFalse(audit["full_coco2017"])
            with self.assertRaisesRegex(ValueError,"full MS COCO"):
                inspect_coco(root)

    def test_missing_image_and_renumbering_are_rejected(self):
        with test_workspace() as temp:
            root = temp / "coco"
            synthetic(root)
            path = root / "annotations/instances_val2017.json"
            data = json.loads(path.read_text())
            data["categories"][-1]["id"] = 89
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"original 80 category IDs"):
                inspect_coco(root,allow_subset=True)

    def test_subset_retains_ids_and_records_provenance(self):
        with test_workspace() as temp:
            source, output = temp/"source",temp/"small"
            synthetic(source)
            audit = subset(source,output,2,1,copy_images=True)
            data = json.loads((output/"annotations/instances_train2017.json").read_text())
            self.assertEqual(audit["splits"]["train"]["images"],2)
            self.assertEqual(data["annotations"][0]["category_id"],90)
            self.assertTrue((output/"subset_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
