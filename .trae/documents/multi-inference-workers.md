# 多推理实例并行方案

## 摘要

将当前单模型单线程推理架构改为多模型实例多线程推理架构，多个推理线程共享同一个 `ready_queue`，谁空闲谁领任务，提升 GPU 利用率和吞吐量。

## 当前状态分析

当前 `DynamicBatcher` 架构：

```
asyncio.Queue → _dispatch_loop → 预处理线程池(4) → ready_queue → GPU推理线程(1个) → 后处理线程池(4)
```

* **1 个模型实例**：`self._model = pipeline(...)` 只创建一次

* **1 个推理线程**：`self._inference_thread` 单线程紧密循环从 `ready_queue` 取任务

* 吞吐瓶颈：单张推理约 50ms，理论极限约 19-20 QPS

* 已设置 `TF_FORCE_GPU_ALLOW_GROWTH=true`，为多实例共享 GPU 显存提供了基础

## 改造方案

### 核心思路

创建 N 个模型实例 + N 个推理线程，所有推理线程共享同一个 `ready_queue`（`queue.Queue` 本身线程安全），谁空闲谁领任务：

```
asyncio.Queue → _dispatch_loop → 预处理线程池(4) → ready_queue ─┬→ 推理线程1(模型实例1) → 后处理线程池(4)
                                                                  ├→ 推理线程2(模型实例2) → 后处理线程池(4)
                                                                  └→ 推理线程N(模型实例N) → 后处理线程池(4)
```

### 具体改动

#### 1. `batch_processor.py` - DynamicBatcher 类改造

**`__init__`** **方法：**

* 新增参数 `num_inference_workers: int = 1`（推理实例数）

* 将 `self._model = None` 改为 `self._models: list = []`（多个模型实例）

* 将 `self._inference_thread` 改为 `self._inference_threads: list = []`（多个推理线程）

**`initialize`** **方法：**

* 循环创建 `num_inference_workers` 个模型实例：

  ```python
  for i in range(self.num_inference_workers):
      print(f"[Batcher] 加载模型实例 {i+1}/{self.num_inference_workers}...")
      model = pipeline(Tasks.universal_matting, model=self.model_path)
      self._models.append(model)
  ```

* 每个模型实例都做预热

* 创建并启动 `num_inference_workers` 个推理线程，每个线程绑定自己的模型实例：

  ```python
  for i, model in enumerate(self._models):
      t = threading.Thread(
          target=self._inference_loop,
          args=(model,),
          name=f"gpu-inference-{i}",
          daemon=True
      )
      self._inference_threads.append(t)
      t.start()
  ```

**`_inference_loop`** **方法：**

* 改为接收 `model` 参数：`def _inference_loop(self, model)`

* 推理调用从 `self._model([...])` 改为 `model([...])`

* 其余逻辑不变（从共享的 `ready_queue` 取任务，提交后处理）

**`shutdown`** **方法：**

* 遍历等待所有推理线程结束：`for t in self._inference_threads: t.join(timeout=5.0)`

**`get_stats`** **方法：**

* 新增返回 `inference_workers` 数量

#### 2. `main.py` - 配置与初始化

**新增环境变量：**

* `NUM_INFERENCE_WORKERS`：推理实例数，默认 `1`（向后兼容）

**lifespan 中传递参数：**

* `DynamicBatcher` 初始化时传入 `num_inference_workers=NUM_INFERENCE_WORKERS`

**启动日志：**

* 打印推理实例数

## 文件改动清单

| 文件                   | 改动内容                                                                                                                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch_processor.py` | `__init__` 新增 `num_inference_workers` 参数，`_model` → `_models` 列表，`_inference_thread` → `_inference_threads` 列表；`initialize` 循环创建多模型多线程；`_inference_loop` 接收 model 参数；`shutdown` 遍历 join 所有线程；`get_stats` 返回 worker 数 |
| `main.py`            | 新增 `NUM_INFERENCE_WORKERS` 环境变量配置，传递给 `DynamicBatcher`，启动日志打印推理实例数                                                                                                                                                   |

## 假设与决策

1. **ModelScope pipeline 支持同进程多实例**：假设 `pipeline()` 可以在同一进程中多次调用创建独立实例。`TF_FORCE_GPU_ALLOW_GROWTH=true` 已确保多实例共享 GPU 显存不会一次性占满。
2. **推理线程安全**：每个模型实例只被一个线程使用，不存在并发访问同一模型的问题。
3. **后处理线程池共享**：所有推理线程共享同一个后处理线程池，因为后处理是纯 CPU 操作，4 个线程足够。
4. **默认值为 1**：不设置环境变量时行为与原来完全一致，向后兼容。

## 验证步骤

1. 设置 `NUM_INFERENCE_WORKERS=1`，验证行为与改造前完全一致
2. 设置 `NUM_INFERENCE_WORKERS=2` 或 `3`，验证多个推理线程正常工作
3. 通过 `/health` 和 `/api/stats` 接口确认 `inference_workers` 数量正确
4. 压力测试对比单实例 vs 多实例的 QPS 提升
5. 监控 GPU 显存使用，确认多实例不会 OOM

