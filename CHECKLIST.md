# 最新提交格式清单

ZIP名称和唯一顶层目录：20242054_黄刘昱泽_课程大作业。按老师2026-10-22追加要求，目标100 MB以内、最大不超过300 MB。

- [x] README.md：内容与入口。
- [x] proposal.pdf：选题、任务、单人100%分工。
- [x] report.pdf：任务、方法、数据、结构、环境、参数、结果、对照、可视化、失败分析、分工、总结与引用。
- [x] presentation.pptx：19页，保留感谢聆听页与逐页备注。
- [x] src/：公开RF-DETR基线源码及许可；不含SATR代码。
- [x] data/：来源链接、60张样例及GT、固定划分；完整标注在线下载。
- [x] checkpoints/：实际模型清单、SHA256、外置下载说明；不包含权重二进制。
- [x] logs/：六组真实文本/JSON日志与6份TensorBoard。
- [x] results/：图表、完整指标、真实样例预测、对照、审计与8张演示。
- [x] scripts/：train.sh、evaluate.sh、run_demo.sh、download_assets.py、verify_submission.py。
- [x] environment.yml、requirements.txt、LICENSE.txt。
- [x] external_assets.json：独立课程仓库在线资源、字节数和SHA256。

不收录SATR论文原稿、完整图像数据集、模型二进制、完整预测、环境和wheel缓存。原始完整资料仅在本地保留；GitHub大文件采用独立课程Release，不改动论文仓库。

- [x] SATR-SSM全部实现、适配/导出/兼容测试和源码附件下载已排除。D/F仅提供原始实验记录及保存预测回放。
