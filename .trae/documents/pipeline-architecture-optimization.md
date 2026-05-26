# 抠图服务高并发架构优化方案

## 问题分析

### 当前架构的核心瓶颈

当前架构采用"收集批次 → 加锁 → 整批推理"模式，存在以下问题：

1. **伪 batch 无收益**：`self._model([path1, path2, ...])` 内部是逐张串行推理，并非 GPU 真正的 batch 并行。传 16 张和传 1 张，GPU 计算量完全相同（16 次串行前向传播），但持锁时间却成倍增长。

2. **`_model_lock` + 伪 batch = 最差组合**：全局锁将所有线程串行化，batch 又让单次持锁时间成倍增长。`NUM_WORKERS=4` 形同虚设——4 个线程排队等一把锁，同一时刻只有 1 个线程在推理。

3. **批次收集增加延迟**：`_batch_loop` 先等 `max_wait_time` 收集任务再处理，用户请求必须等待凑批次，增加了不必要的排队延迟。

4. **临时文件 I/O 在锁内**：图片解码、保存临时文件、结果编码都在 `_model_lock` 内执行，GPU 在这些 CPU 操作期间是空闲的。

5. **GPU 利用率低**：GPU 推理是串行的，但当前架构没有做到"GPU 推理时 CPU 同时准备下一张图"，导致 GPU 在两次推理之间有空闲间隙。

### 单 GPU + 串行模型的最优策略

既然模型只能逐张推理，核心目标就是：**让 GPU 推理之间零间隙，CPU 工作与 GPU 工作重叠执行**。

## 新架构：流水线（Pipeline）+ 预取（Pre-fetch）

### 架构图

```
HTTP 请求
   │
   ▼
asyncio.Queue (提交任务)
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  _dispatch_loop (asyncio 协程)                               │
│  从 asyncio.Queue 取任务 → 提交到预处理线程池                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  预处理线程池 (4 线程)                                        │
│  解码 bytes → PIL Image → 保存临时 BMP 文件                   │
│  完成后放入 ready_queue                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ready_queue (thread-safe)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  GPU 推理线程 (1 个专用线程，无锁)                             │
│  紧密循环：取任务 → model([path]) → 提交后处理                 │
│  ★ GPU 永远不需要等待 CPU ★                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  后处理线程池 (4 线程)                                        │
│  裁剪透明边缘 → cv2.imencode → 删除临时文件                    │
│  loop.call_soon_threadsafe(future.set_result, result)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              asyncio 事件循环唤醒等待的协程
                       │
                       ▼
                 HTTP 响应返回给用户
```

### 关键设计点

#### 1. 三级流水线，CPU/GPU 重叠

| 阶段 | 执行者 | 耗时（估算） | 是否阻塞 GPU |
|------|--------|-------------|-------------|
| 预处理 | CPU 线程池 | ~5-10ms | 否，与 GPU 并行 |
| GPU 推理 | 专用线程 | ~30-80ms | 是（不可避免） |
| 后处理 | CPU 线程池 | ~5-10ms | 否，与 GPU 并行 |

时间线示意：
```
时间 →  0ms    10ms    40ms    50ms    80ms    90ms   120ms
预处理:  [img1]          [img2]          [img3]
GPU:            [===img1===]   [===img2===]   [===img3===]
后处理:                  [img1]          [img2]          [img3]
```

GPU 推理之间几乎零间隙（仅 `ready_queue.get()` 的微秒级开销）。

#### 2. 单推理线程，无锁

- 只有一个线程访问 GPU，不需要 `_model_lock`
- 消除锁竞争和线程等待
- 推理线程逻辑极简：取任务 → 推理 → 提交后处理 → 循环

#### 3. 预取机制

- 预处理线程池提前将图片解码并保存为临时文件
- GPU 推理线程从 `ready_queue` 取到的任务已经准备好，直接推理
- 当 GPU 在处理第 N 张图时，第 N+1 张的临时文件可能已经写好

#### 4. 临时文件使用 BMP 格式

- BMP 无压缩，写入速度远快于 PNG（省去压缩耗时）
- 文件稍大但临时文件生命周期短，磁盘空间不是问题
- ModelScope pipeline 内部用 cv2.imread 读取，完全支持 BMP

#### 5. 线程安全的 Future 解析

- 后处理线程通过 `loop.call_soon_threadsafe(future.set_result, result)` 安全地唤醒 asyncio 协程
- 不需要 `run_in_executor` 的额外开销

## 具体修改方案

### 文件 1: `batch_processor.py` — 重写核心架构

#### 1.1 `MattingTask` 增加字段

```python
@dataclass
class MattingTask:
    task_id: str
    image_data: bytes
    image_format: str
    future: asyncio.Future
    created_at: float
    trim_edges: bool = True
    # 新增字段
    temp_path: Optional[str] = None      # 预处理写入的临时文件路径
    loop: Optional[asyncio.AbstractEventLoop] = None  # 事件循环引用，用于线程安全回调
```

#### 1.2 `DynamicBatcher` 重写

**删除的内容：**
- `_model_lock` — 不再需要，单推理线程无竞争
- `_batch_loop()` — 替换为 `_dispatch_loop()`
- `_process_batch()` — 替换为流水线处理
- `_inference_batch()` — 替换为单张推理

**新增的内容：**

| 成员 | 类型 | 用途 |
|------|------|------|
| `_ready_queue` | `queue.Queue` | 预处理完成 → 推理线程的消费队列（线程安全） |
| `_preprocess_pool` | `ThreadPoolExecutor(4)` | 预处理线程池 |
| `_postprocess_pool` | `ThreadPoolExecutor(4)` | 后处理线程池 |
| `_inference_thread` | `threading.Thread` | 专用 GPU 推理线程 |
| `_loop` | `asyncio.AbstractEventLoop` | 事件循环引用 |

**新增方法：**

| 方法 | 运行环境 | 职责 |
|------|---------|------|
| `_dispatch_loop()` | asyncio 协程 | 从 asyncio.Queue 取任务，提交预处理 |
| `_preprocess(task)` | 预处理线程池 | 解码图片 + 保存 BMP 临时文件 → 放入 ready_queue |
| `_inference_loop()` | 专用线程 | 紧密循环：取任务 → model 推理 → 提交后处理 |
| `_postprocess(task, raw_result)` | 后处理线程池 | 裁剪 + 编码 + 删临时文件 + set_result |

**`initialize()` 修改：**
- 启动 `_dispatch_loop` 协程
- 启动 `_inference_thread` 线程
- 不再启动 `_batch_loop`

**`shutdown()` 修改：**
- 设置 `_shutdown` 标志
- 等待推理线程结束
- 关闭预处理和后处理线程池

**`submit()` 不变** — 仍然 `put_nowait` 到 asyncio.Queue

**`get_stats()` 不变** — 统计逻辑保持

### 文件 2: `main.py` — 调整配置和修复

#### 2.1 配置参数调整

```python
# 删除 MAX_BATCH_SIZE（不再有意义）
# 删除 MAX_WAIT_TIME（不再有批次收集等待）

MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "500"))       # 200 → 500，缓冲突发流量
PREPROCESS_WORKERS = int(os.getenv("PREPROCESS_WORKERS", "4")) # 新增：预处理线程数
POSTPROCESS_WORKERS = int(os.getenv("POSTPROCESS_WORKERS", "4")) # 新增：后处理线程数
```

#### 2.2 DynamicBatcher 构造调用调整

```python
batcher = DynamicBatcher(
    model_path=MODEL_PATH,
    max_queue_size=MAX_QUEUE_SIZE,
    preprocess_workers=PREPROCESS_WORKERS,
    postprocess_workers=POSTPROCESS_WORKERS,
)
```

#### 2.3 修复弃用 API

```python
# 旧：asyncio.get_event_loop().create_future()
# 新：asyncio.get_running_loop().create_future()
```

涉及行：main.py 中所有 `asyncio.get_event_loop().create_future()` 调用（约 3 处）

#### 2.4 启动日志调整

更新打印的配置信息，移除 `MAX_BATCH_SIZE` 和 `MAX_WAIT_TIME`，新增 `PREPROCESS_WORKERS` 和 `POSTPROCESS_WORKERS`。

#### 2.5 `/health` 和 `/api/stats` 路由调整

- `queue_size` 改为同时报告 `_queue` 和 `_ready_queue` 的大小
- 配置信息中移除 `max_batch_size` 和 `max_wait_time`

### 文件 3: `stress_test.py` — 无需修改

压力测试脚本通过 HTTP 调用 API，不依赖内部实现，无需修改。

## 吞吐量估算

假设单张 GPU 推理耗时 50ms：

| 架构 | GPU 利用率 | 理论吞吐 | 说明 |
|------|-----------|---------|------|
| 当前（伪 batch + 锁） | ~60% | ~12 QPS | GPU 空闲等待 CPU 预处理、锁竞争 |
| 新架构（流水线 + 预取） | ~95% | ~19 QPS | GPU 几乎无空闲，CPU 工作完全重叠 |

单 GPU 单进程极限约 19-20 QPS（取决于单张推理速度）。要达到 100 QPS，需要：
- **多进程部署**：`uvicorn workers=6`（每个 worker 独立模型实例，需要足够 GPU 显存）
- 或 **模型加速**：TensorRT/ONNX 量化，将单张推理从 50ms 降到 10ms

## 验证步骤

1. 启动服务，确认模型加载和预热正常
2. 单张请求测试，确认返回结果正确
3. 使用 `stress_test.py --rate 10 --duration 30` 测试 10 QPS 场景
4. 使用 `stress_test.py --rate 20 --duration 30` 测试极限场景
5. 对比优化前后的 P50/P90/P99 延迟和吞吐量
6. 检查 GPU 利用率（`nvidia-smi`）是否接近 100%
