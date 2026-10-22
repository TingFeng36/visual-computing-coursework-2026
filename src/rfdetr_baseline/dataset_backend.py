from coco_tools import inspect_coco
from data_tools import validate_dataset


def audit_dataset(kind, root, *, validation_only=False, allow_coco_subset=False):
    if kind == "coco":
        return inspect_coco(root, ("valid",) if validation_only else ("train","valid"), allow_subset=allow_coco_subset)
    if kind != "roboflow":
        raise ValueError("dataset_file must be coco or roboflow")
    print(f"[M3FD/custom check] checking annotations and image files: {root}", flush=True)
    result = validate_dataset(root, verify_pixels=True)
    print("[M3FD/custom check] passed", flush=True)
    result.update(dataset_file="roboflow", num_classes=len(result["classes"]),
                  category_names={i+1:name for i,name in enumerate(result["classes"])})
    return result
