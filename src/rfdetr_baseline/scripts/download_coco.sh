#!/usr/bin/env bash
set -euo pipefail
COCO_TASK_ROOT="${1:-/root/autodl-tmp/datasets/coco2017}"
COCO_TASK_ARCHIVES="$COCO_TASK_ROOT/archives"
mkdir -p "$COCO_TASK_ARCHIVES"
for COCO_TASK_FILE in train2017.zip val2017.zip annotations_trainval2017.zip; do
  if [[ "$COCO_TASK_FILE" == annotations* ]]; then
    COCO_TASK_SECTION="annotations"
  else
    COCO_TASK_SECTION="zips"
  fi
  wget -c --tries=10 --timeout=30 \
    "https://s3.amazonaws.com/images.cocodataset.org/$COCO_TASK_SECTION/$COCO_TASK_FILE" \
    -O "$COCO_TASK_ARCHIVES/$COCO_TASK_FILE"
done
COCO_TASK_ROOT="$COCO_TASK_ROOT" python - <<'PY'
import os, json, hashlib
from pathlib import Path
root = Path(os.environ['COCO_TASK_ROOT']).resolve()
sizes = {'train2017.zip': 19336861798, 'val2017.zip': 815585330,
         'annotations_trainval2017.zip': 252907541}
records = {}
for name, size in sizes.items():
    path = root / 'archives' / name
    if path.stat().st_size != size:
        raise SystemExit(f'Unexpected size: {name}; do not extract an incomplete/changed archive')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    section = 'annotations' if name.startswith('annotations') else 'zips'
    records[name] = {'bytes':size,'observed_sha256':digest.hexdigest(),
                     'source_url':f'https://s3.amazonaws.com/images.cocodataset.org/{section}/{name}'}
(root/'download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
PY
for COCO_TASK_FILE in train2017.zip val2017.zip annotations_trainval2017.zip; do
  unzip -tq "$COCO_TASK_ARCHIVES/$COCO_TASK_FILE"
  unzip -nq "$COCO_TASK_ARCHIVES/$COCO_TASK_FILE" -d "$COCO_TASK_ROOT"
done
echo "MS COCO 2017 extracted at $COCO_TASK_ROOT; existing files were preserved."
