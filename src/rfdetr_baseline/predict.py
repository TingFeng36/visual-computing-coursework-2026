from __future__ import annotations
import argparse
from pathlib import Path
import json
from common import build_nano, write_json
from dataset_backend import audit_dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--dataset-file",choices=["roboflow","coco"],default="roboflow")
    parser.add_argument("--allow-coco-subset",action="store_true")
    parser.add_argument("--weights",required=True)
    parser.add_argument("--image",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--threshold",type=float,default=.3)
    args=parser.parse_args()
    from PIL import Image,ImageDraw
    audit=audit_dataset(args.dataset_file,args.dataset,validation_only=True,allow_coco_subset=args.allow_coco_subset)
    detector=build_nano(args.weights,audit["num_classes"],checkpointing=False)
    image=Image.open(args.image).convert("RGB")
    detections=detector.predict(image,threshold=args.threshold)
    draw=ImageDraw.Draw(image)
    records=[]
    for box,score,category in zip(detections.xyxy,detections.confidence,detections.class_id):
        category=int(category)
        if category not in audit["category_names"]:
            continue
        name=audit["category_names"][category]
        box=list(map(float,box))
        draw.rectangle(box,outline="lime",width=2)
        draw.text((box[0],max(0,box[1]-12)),f"{name} {score:.2f}",fill="lime")
        records.append({"category_id":category,"class":name,"score":float(score),"xyxy":box})
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    image.save(output)
    write_json(output.with_suffix(".json"),records)
    print("Saved",output)


if __name__=="__main__":
    main()
