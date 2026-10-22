# 原始训练日志

A/B/C/D各20轮，E/F各100轮，共280轮。logs/A到F中的train.log、log.txt及验证日志均与原始归档字节一致。

为避免重复提交旧版拼接日志，根目录要求的logs/train.log使用F组实际原始train.log副本；六组完整日志分别保留。logs/val.log保留六组最终验证记录，logs/tensorboard保留6份原始事件文件。
