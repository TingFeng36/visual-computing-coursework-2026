import json
import unittest
from PIL import Image
from test_data import test_workspace
from prepare_ourmamba_m3fd import prepare


class ExistingM3fdTests(unittest.TestCase):
    def fixture(self,root,has_test=False):
        for split in ("train","test"):
            images,labels=root/"images"/split,root/"labels"/split
            images.mkdir(parents=True);labels.mkdir(parents=True)
            Image.new("RGB",(80,40),"red" if split=="train" else "blue").save(images/"one.png")
            (labels/"one.txt").write_text("1 .5 .5 .2 .3\n")
        hidden=root/"images/test/.ipynb_checkpoints"
        hidden.mkdir()
        Image.new("RGB",(80,40)).save(hidden/"one-checkpoint.png")
        source=root/"m3fd.yaml"
        source.write_text("train: images/train\nval: images/test\nnc: 2\nnames: [original_first, original_last]\n"+
                          ("test: images/test\n" if has_test else ""))
        return source

    def test_original_classes_and_hidden_checkpoint_exclusion(self):
        with test_workspace() as root:
            source=self.fixture(root/"source")
            result=prepare(source,root/"out",copy_images=True)
            self.assertEqual(result["classes"],["original_first","original_last"])
            self.assertEqual(result["splits"]["valid"]["images"],1)
            self.assertTrue(result["test_is_validation_alias"])
            anns=json.loads((root/"out/train/_annotations.coco.json").read_text())
            self.assertEqual(anns["annotations"][0]["category_id"],2)

    def test_same_val_test_is_marked_as_alias(self):
        with test_workspace() as root:
            source=self.fixture(root/"source",has_test=True)
            result=prepare(source,root/"out",copy_images=True)
            self.assertTrue(result["test_is_validation_alias"])
            manifest=json.loads((root/"out/dataset_manifest.json").read_text())
            self.assertEqual(manifest["original_source"]["validation_origin"],"original val")

    def test_missing_validation_is_not_randomly_resplit(self):
        with test_workspace() as root:
            source=self.fixture(root/"source")
            source.write_text("train: images/train\nnames: [original_first, original_last]\n")
            with self.assertRaisesRegex(ValueError,"neither val nor test"):
                prepare(source,root/"out",inspect_only=True)


if __name__ == '__main__':
    unittest.main()
