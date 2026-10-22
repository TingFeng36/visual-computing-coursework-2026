"""Download the exact URL declared by upstream 1.3.0 and record its SHA256."""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from common import sha256, write_json

URL = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
EXPECTED_SHA256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"


def verify_coco_checkpoint(path):
    import json
    path = Path(path)
    manifest = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError("Official weight or provenance manifest missing; run download_weights.py")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    actual = sha256(path)
    if actual != EXPECTED_SHA256 or actual != record["sha256"] or record["source_url"] != URL:
        raise ValueError("Official COCO checkpoint hash/provenance mismatch")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/rf-detr-nano.pth")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(".manifest.json")
    if path.is_file():
        if not manifest.is_file():
            raise SystemExit("Existing checkpoint has no provenance manifest; choose a new output path.")
        verify_coco_checkpoint(path)
        print("Existing verified checkpoint:", path)
        return
    partial = path.with_suffix(path.suffix + ".part")
    with urlopen(Request(URL, headers={"User-Agent": "RF-DETR-reproduction"}), timeout=120) as response:
        length = response.headers.get("Content-Length")
        with partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    if length is not None and partial.stat().st_size != int(length):
        raise SystemExit("Incomplete checkpoint download; final file was not published.")
    if sha256(partial) != EXPECTED_SHA256:
        raise SystemExit("Official weight SHA256 changed; final file was not published. Verify upstream deliberately.")
    partial.replace(path)
    record = {"source_url": URL, "source_reference": "rf-detr 1.3.0 rfdetr/main.py HOSTED_MODELS",
              "bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(manifest, record)
    print(record)


if __name__ == "__main__":
    main()
