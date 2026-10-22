# 真实样例预测与完整预测

各组predictions.sample.coco.json为真实最终预测按包内图像ID的精确筛选，保留全部原预测条目和分数，不做阈值截断。sample_manifest.json记录来源、完整文件SHA256、图像ID和条目数。C/D、E/F用于离线演示。

报告中的AP基于完整验证集。完整预测在独立课程GitHub Release外置，恢复命令：

```bash
python scripts/download_assets.py --keys predictions
```

恢复后各组完整文件名为predictions.coco.json，已有绘图入口会优先使用完整文件。此前完整预测经COCOeval复算的72项指标保存在results/comparisons/prediction_recomputation.json。
