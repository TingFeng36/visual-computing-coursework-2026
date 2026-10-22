# 数据说明

## 来源

MS COCO与COCO为同一数据集。本项目使用COCO2017 train2017（118287张）和val2017（5000张）。原始下载入口：https://cocodataset.org/#download 。数据图像版权归原图像持有人，使用范围遵循COCO原始数据条款。

M3FD使用已有可见光分支和既定train/test目录，不使用红外融合。项目原始数据目录为 datasets_M3FD/images/train 与 images/test；转换后对应train与valid，各2100张。类别依次为People、Car、Bus、Motorcycle、Lamp、Truck。数据来源为M3FD原项目：https://github.com/JinyuanLiu-CV/TarDAL 。本包保留样例GT和来源路径清单；完整固定标注从课程Release恢复，原图从数据提供方获取。

## 划分与标签空间

- A：COCO全量118287张训练，5000张验证，80个实际类别，原始稀疏类别ID布局。
- B：固定seed=0，按SHA256(seed:image_id)选择5000张训练图，保留80类；验证沿用全部val2017。
- C/D：与B完全相同的5000张训练图与验证图，保留VOC对应20类；被移除的60类目标按该任务定义作为背景。
- VOC20原COCO类别ID为：1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72，稠密映射和图像ID见manifests/paired_subset_manifest.json。
- E/F：M3FD固定训练/验证划分，train=2100、valid=2100；类别顺序保持原YOLO YAML。
- COCO20与M3FD的test是valid别名；本次报告全部使用valid，不将它记为独立测试集。

splits/*.jsonl逐行保存实际image_id与file_name；固定任务完整标注已外置至Release，下载后恢复到annotations/。COCO2017全量训练标注从官方annotations_trainval2017.zip恢复，哈希见对应dataset_audit.json；本包不附完整训练图像或整份大规模训练标注。

## 少量样例

samples/coco20_voc_5k有47张图片及对应GT，samples/m3fd有13张图片及GT，共60张。它们用于示例回放与失败案例分析，完整验证指标来自原始最终验证记录及全量预测复算；包内results/predictions只保留这些样例图像的真实预测。

## 恢复可训练目录

先执行`python scripts/download_assets.py --keys annotations`恢复固定标注，再把COCO整理为train2017/、val2017/、annotations/；M3FD可使用原始images/train、images/test，或已转换的train/valid目录。

```bash
python scripts/prepare_data.py \
  --coco-root /datasets/COCO2017 \
  --m3fd-root /datasets/datasets_M3FD \
  --output /datasets/course
```

默认用软链接连接原图，使用--copy可复制。脚本从提交的原标注恢复B/C/D/E/F的固定划分，不重新随机抽样。A训练直接指向完整COCO根目录。数据清单保存的历史服务器路径用于追溯，脚本会按图像名称映射当前输入目录。

## 可验证的原始数据入口

- COCO官方页面：https://cocodataset.org/#download
- train2017：https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip
- val2017：https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
- COCO标注：https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
- M3FD作者项目：https://github.com/JinyuanLiu-CV/TarDAL#download
- M3FD Google Drive：https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6
- M3FD百度网盘：https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD （提取码M3FD）

原提供方默认划分与课程历史划分可能不同；复现本次结果使用本包固定清单及Release标注，不重新随机划分。
