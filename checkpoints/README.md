# 外置模型下载

按老师最新体积要求，提交包不含任何.pt/.pth模型二进制。六组实际训练的最佳权重分别发布于独立课程仓库的Release：

https://github.com/TingFeng36/visual-computing-coursework-2026/releases/tag/course-20261022-v1

checkpoint_manifest.json保留六组权重的原始字节数、SHA256与模型参数量；external_assets.json提供可直接下载的文件链接。它们是训练实际生成的最佳模型，未重新训练或重新序列化。

| 组 | 下载键 | 下载后恢复路径 | 参数量M |
|---|---|---|---:|
| A | A | checkpoints/A/best_model.pth | 30.467 |
| B | B | checkpoints/B/best_model.pth | 30.467 |
| C | C | checkpoints/C/best_model.pth | 30.215 |
| D | D | checkpoints/D/best_model.pth | 9.577 |
| E | E | checkpoints/E/best_model.pth | 30.467 |
| F | F | checkpoints/best_model.pth | 9.829 |

```bash
python scripts/download_assets.py --keys C,initialization
```

全部六组模型下载完成后，可再执行`python scripts/verify_submission.py --downloaded-models`。

模型核验命令在六组模型均已下载时检查全部权重。只下载指定组时由下载脚本核验其SHA256。

官方COCO初始化直接从原提供方获取（非本次最佳权重）：
https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth

原始完整SATR-SSM与本次D/F骨干迁移模型分别命名；原完整模型0.435/0.706的M3FD结果按既有协议单列。最佳模型不含完整优化器状态。

源码排除版：不包含SATR导出目录及其下载。D/F最佳模型为张量权重，不含Python源码；此提交不提供D/F模型的训练或重新推理实现。可通过保存预测核对结果。
