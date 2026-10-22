# 视觉计算基础与应用课程大作业

**SATR-SSM骨干的跨检测器迁移验证**。学号20242054，黄刘昱泽，单人完成100%。

本版按老师2026-10-22追加要求整理：提交ZIP目标小于100 MB，严格低于300 MB；模型、完整预测与全量任务标注通过在线附件下载。完整原始归档保留在本地。

- 独立课程仓库：https://github.com/TingFeng36/visual-computing-coursework-2026
- 提交包与外置附件：https://github.com/TingFeng36/visual-computing-coursework-2026/releases/tag/course-20261022-v1
- 所有下载地址、字节数与SHA256：external_assets.json

## 提交内容

| 路径 | 内容 |
|---|---|
| proposal.pdf / report.pdf / presentation.pptx | 选题分工、13页报告、19页正式PPT（含感谢聆听） |
| src/ | RF-DETR基线训练/评估代码和固定上游源码；不含SATR实现 |
| data/ | 数据下载入口、60张原始样例、样例GT、固定划分和类别映射 |
| checkpoints/ | 六组模型清单、原始SHA256、结构元数据与下载说明；无权重二进制 |
| logs/ | 六组未改写的真实文本/逐轮JSON日志及6份TensorBoard；根train.log为F组原日志副本 |
| results/ | 全集指标表、训练曲线、同图可视化、8张演示、真实样例预测与审计 |
| scripts/ | A/B/C/E基线训练评估、六组保存预测回放、下载和校验入口 |

## 实验结果

| 组 | 模型 + Backbone | 数据集 | 参数量M | mAP50:95 | mAP50 |
|---|---|---|---:|---:|---:|
| A | RF-DETR-Nano + DINOv2-S | COCO2017全量/80类 | 30.47 | 0.483 | 0.674 |
| B | RF-DETR-Nano + DINOv2-S | COCO2017 5k/80类 | 30.47 | 0.483 | 0.673 |
| C | RF-DETR-Nano + DINOv2-S | COCO2017 5k/VOC20类 | 30.22 | 0.544 | 0.748 |
| D | RF-DETR-Nano + SATR-SSM骨干 | COCO2017 5k/VOC20类 | 9.58 | 0.006 | 0.017 |
| E | RF-DETR-Nano + DINOv2-S | M3FD可见光 | 30.47 | 0.493 | 0.782 |
| F | RF-DETR-Nano + SATR-SSM骨干 | M3FD可见光 | 9.83 | 0.315 | 0.561 |

B/C 统一在共同20类上比较时，mAP50:95为0.543与0.544。原始完整SATR-SSM检测器为3.72M，M3FD结果为0.435/0.706，采用原论文协议，在报告中另列来源。

模型初始化使用已有预训练权重；A为本课程20轮全量微调协议，NAS搜索和原论文全部训练并未纳入本次实验。C/D与E/F同时更换编码器与投影接口。基线训练为Python3.11，SATR训练为Python3.12，报告记录了该环境差异和同权重推理回归。全部本次数值为验证集结果；test目录对应验证集别名。

## 不下载模型即可检查

```bash
python scripts/verify_submission.py
python scripts/render_demo.py --output runs/demo_replay --limit 4
tensorboard --logdir logs/tensorboard
```

样例预测从原始完整预测中按包内图片ID逐条提取，未重新推理。报告AP来自完整验证集，不能由样例重新计算。

## 下载与复现

```bash
python scripts/download_assets.py --list
# 恢复实际实验使用的固定标注和训练初始化
python scripts/download_assets.py --keys annotations,initialization
# 下载基线C组权重
python scripts/download_assets.py --keys C
# 如需重算六组完整AP，另行获取全部保存预测
python scripts/download_assets.py --keys predictions
```

下载脚本按SHA256校验并恢复原路径。COCO和M3FD完整图片从data/README.md的原提供方获取；再按docs/复现说明.md建立环境并恢复固定任务。

```bash
bash scripts/train.sh --group C --dataset /datasets/course/coco20_voc_5k --output /runs/C_reproduce
bash scripts/evaluate.sh --group C --dataset /datasets/course/coco20_voc_5k --output /runs/C_valid
```

## 格式与证据

ZIP及唯一顶层目录为20242054_黄刘昱泽_课程大作业。README、proposal、report、PPTX、源码、日志、TensorBoard、结果、演示、脚本和许可均保留。按最新要求以下载清单替代包内权重及完整数据。

所有mAP按0–1的三位小数展示；六组280轮日志和正式结果保持原值。本版以FILE_MANIFEST.json、SHA256SUMS.txt和results/comparisons/source_exclusion_audit.json为准。

原始完整SATR-SSM为SATR骨干+PAN-FPN+原检测头；D/F为SATR骨干+新适配层+RF-DETR-Nano解码器/检测头。仅保留SATR pipeline图，不收录论文原稿。论文项目仓库不承载本课程材料。

## SATR-SSM源码排除

本提交与课程仓库不提供任何SATR-SSM骨干、模块、路由器、适配层、兼容补丁、训练/导出实现或相关测试源码，也不提供SATR导出源码下载。A/B/C/E可由随附RF-DETR基线源码训练与评估；D/F保留实际日志、配置、权重下载、指标和保存预测，用于结果核对与CPU回放，提交包不提供其训练或重新推理实现。原始SATR论文稿不收录，保留pipeline图。
