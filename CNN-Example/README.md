### profile_resnet_tasks.py
ResNet 任务采样代码，会将每个子任务（如前向传播、反向传播、优化器更新等）的性能数据写入 resnet_task_profiles.csv，每一行代表一次任务阶段的采样结果。以下是该CSV每一列的解释：
```angular2html
| 列名                 | 含义                                                                                                  | 示例                |
| ------------------ | --------------------------------------------------------------------------------------------------- | ----------------- |
| `task_name`        | 子任务名称，例如：`forward_pass`（前向传播）、`loss_computation`（损失计算）、`backward_pass`（反向传播）、`optimizer_step`（参数更新） | `forward_pass`    |
| `duration_sec`     | 子任务耗时，单位为秒，反映执行该子任务所需时间                                                                             | `0.0234`          |
| `gpu_mem_MB`       | GPU显存使用量（相对增长），单位是MB，表示执行该子任务过程中额外使用了多少显存                                                           | `95.0`            |
| `gpu_util_percent` | GPU计算单元使用率（百分比），采样点时刻的GPU利用率                                                                        | `76`              |
| `batch_size`       | 本轮处理的样本数，即批次大小                                                                                      | `32`              |
| `input_shape`      | 输入张量的形状，通常表示`(batch_size, channels, height, width)`                                                 | `(32, 3, 32, 32)` |
| `flops_estimate`   | 任务对应的浮点计算量估计（单位：GFLOPs），仅在 `forward_pass` 时估算一次                                                     | `0.55`            |

```
