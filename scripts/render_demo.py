"""CPU visualization of authentic saved predictions; no retraining or inferred metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'results/demo')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--threshold', type=float, default=.3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    receipt = []
    for dataset, baseline, satr in [('coco20_voc_5k','C','D'),('m3fd','E','F')]:
        base = ROOT/'data/samples'/dataset
        gt = json.loads((base/'_annotations.coco.json').read_text(encoding='utf-8'))
        images = sorted(gt['images'], key=lambda r:r['id'])[:args.limit]
        selected = {r['id'] for r in images}
        cats = {r['id']:r['name'] for r in gt['categories']}
        detections = {}
        for group in [baseline, satr]:
            prediction_path = ROOT/f'results/predictions/{group}/predictions.coco.json'
            if not prediction_path.is_file():
                prediction_path = ROOT/f'results/predictions/{group}/predictions.sample.coco.json'
            rows = json.loads(prediction_path.read_text(encoding='utf-8'))
            mapping = defaultdict(list)
            for row in rows:
                if row['image_id'] in selected and row['score'] >= args.threshold:
                    mapping[row['image_id']].append(row)
            detections[group] = mapping
            del rows
        for image in images:
            raw = Image.open(base/image['file_name']).convert('RGB')
            w, h = 640, round(raw.height*640/raw.width)
            result = Image.new('RGB',(w*3,h+92),'white')
            rows_gt = [r for r in gt['annotations'] if r['image_id'] == image['id']]
            sources = [('GT',rows_gt,(20,160,70)),('DINOv2-S',detections[baseline][image['id']],(32,109,197)),('SATR-SSM',detections[satr][image['id']],(198,45,53))]
            draw = ImageDraw.Draw(result)
            for index,(title,rows,color) in enumerate(sources):
                x0=index*w
                result.paste(raw.resize((w,h)),(x0,42))
                draw.text((x0+18,10),f'{title}: {len(rows)} boxes',font=font,fill=color)
                for row in rows:
                    x,y,bw,bh=row['bbox']; sx=w/raw.width; sy=h/raw.height
                    box=[x0+x*sx,42+y*sy,x0+(x+bw)*sx,42+(y+bh)*sy]
                    draw.rectangle(box,outline=color,width=3)
            draw.text((18,h+55),f'{dataset} | image_id={image["id"]} | score threshold={args.threshold} | saved predictions',font=font,fill=(65,65,65))
            filename=f'{dataset}_{image["id"]}.png'
            result.save(args.output/filename)
            receipt.append({'dataset':dataset,'image_id':image['id'],'file':filename,'score_threshold':args.threshold,
                            'baseline_group':baseline,'satr_group':satr,'gt_boxes':len(rows_gt),
                            'baseline_displayed_boxes':len(sources[1][1]),'satr_displayed_boxes':len(sources[2][1])})
    (args.output/'demo_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Rendered {len(receipt)} saved-prediction examples; no training or evaluation was run')


if __name__ == '__main__':
    main()
