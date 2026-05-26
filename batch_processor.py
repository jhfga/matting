"""
抠图处理器 - 流水线架构
预处理线程池 → 专用GPU推理线程 → 后处理线程池
CPU 工作与 GPU 推理重叠执行，最大化 GPU 利用率
"""
import asyncio
import io
import os
import queue
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


@dataclass
class MattingTask:
    """单个抠图任务"""
    task_id: str
    image_data: bytes
    image_format: str
    future: asyncio.Future
    created_at: float
    trim_edges: bool = True
    # 流水线内部字段
    temp_path: Optional[str] = None
    loop: Optional[asyncio.AbstractEventLoop] = None


class DynamicBatcher:
    """
    流水线批处理器
    - 预处理线程池：解码图片 + 保存临时文件（与 GPU 并行）
    - 专用推理线程：单张推理，紧密循环（GPU 零间隙）
    - 后处理线程池：裁剪 + 编码 + 返回结果（与 GPU 并行）
    """

    def __init__(
        self,
        model_path: str,
        max_queue_size: int = 500,
        preprocess_workers: int = 4,
        postprocess_workers: int = 4,
        num_inference_workers: int = 1,
    ):
        self.model_path = model_path
        self.max_queue_size = max_queue_size
        self.preprocess_workers = preprocess_workers
        self.postprocess_workers = postprocess_workers
        self.num_inference_workers = num_inference_workers

        # 任务队列（asyncio 端，API 层提交）
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        # 预处理完成后的就绪队列（线程安全，推理线程消费）
        self._ready_queue: queue.Queue = queue.Queue()

        self._shutdown = False

        # 多个模型实例，每个推理线程绑定一个
        self._models: list = []

        # 线程池
        self._preprocess_pool = ThreadPoolExecutor(
            max_workers=preprocess_workers, thread_name_prefix="preprocess"
        )
        self._postprocess_pool = ThreadPoolExecutor(
            max_workers=postprocess_workers, thread_name_prefix="postprocess"
        )

        # 多个专用 GPU 推理线程
        self._inference_threads: list = []

        # 事件循环引用（用于线程安全的 future 回调）
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 统计信息
        self._stats = {
            'total_processed': 0,
            'avg_inference_time': 0.0,
            'avg_total_time': 0.0,
        }
        self._stats_lock = threading.Lock()

    def initialize(self):
        """初始化模型（在事件循环线程中调用）"""
        self._loop = asyncio.get_running_loop()

        # 加载多个模型实例
        for i in range(self.num_inference_workers):
            print(f"[Batcher] 正在加载模型实例 {i+1}/{self.num_inference_workers}: {self.model_path}")
            model = pipeline(Tasks.universal_matting, model=self.model_path)
            self._models.append(model)
            print(f"[Batcher] 模型实例 {i+1} 加载完成")

        # 预热每个模型实例
        print("[Batcher] 预热模型...")
        for i, model in enumerate(self._models):
            try:
                dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
                dummy_path = os.path.join(tempfile.gettempdir(), f"warmup_{i}.bmp")
                cv2.imwrite(dummy_path, dummy_img)
                model([dummy_path])
                os.remove(dummy_path)
            except Exception as e:
                print(f"[Batcher] 模型实例 {i+1} 预热警告: {e}")
        print(f"[Batcher] {self.num_inference_workers} 个模型实例预热完成")

        # 启动调度协程
        asyncio.create_task(self._dispatch_loop())

        # 启动多个推理线程，每个绑定自己的模型实例
        for i, model in enumerate(self._models):
            t = threading.Thread(
                target=self._inference_loop,
                args=(model,),
                name=f"gpu-inference-{i}",
                daemon=True
            )
            self._inference_threads.append(t)
            t.start()
        print(f"[Batcher] 流水线启动完成，{self.num_inference_workers} 个推理线程就绪")

    async def submit(self, task: MattingTask) -> bool:
        """提交任务到队列"""
        try:
            self._queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            return False

    # ========================================================================
    # 调度协程：从 asyncio 队列取任务，提交预处理
    # ========================================================================
    async def _dispatch_loop(self):
        """调度循环：取任务 → 提交预处理线程池"""
        while not self._shutdown:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            # 记录事件循环引用，用于后处理时线程安全回调
            task.loop = self._loop
            # 提交到预处理线程池
            self._preprocess_pool.submit(self._preprocess, task)

    # ========================================================================
    # 预处理：解码图片 + 保存临时 BMP 文件
    # ========================================================================
    def _preprocess(self, task: MattingTask):
        """预处理：解码 bytes → 保存 BMP 临时文件 → 放入就绪队列"""
        try:
            img = Image.open(io.BytesIO(task.image_data)).convert('RGB')
            # 使用 BMP 格式，无压缩，写入速度远快于 PNG
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"matting_{task.task_id}_{int(time.time() * 1000)}.bmp"
            )
            img.save(temp_path, format='BMP')
            task.temp_path = temp_path
            self._ready_queue.put(task)
        except Exception as e:
            # 预处理失败，直接返回错误
            result = {'success': False, 'error': f'预处理失败: {e}'}
            if task.loop and not task.future.done():
                task.loop.call_soon_threadsafe(task.future.set_result, result)

    # ========================================================================
    # GPU 推理线程：紧密循环，单张推理
    # ========================================================================
    def _inference_loop(self, model):
        """专用 GPU 推理线程：取任务 → 推理 → 提交后处理"""
        while not self._shutdown:
            try:
                task = self._ready_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            inference_start = time.time()

            try:
                # 单张推理
                results = model([task.temp_path])
                raw_result = results[0]
                inference_time = time.time() - inference_start

                # 提交后处理到线程池
                self._postprocess_pool.submit(
                    self._postprocess, task, raw_result, inference_time
                )

            except Exception as e:
                inference_time = time.time() - inference_start
                result = {'success': False, 'error': f'推理失败: {e}'}
                if task.loop and not task.future.done():
                    task.loop.call_soon_threadsafe(task.future.set_result, result)
                self._update_stats(inference_time, 0)

    # ========================================================================
    # 后处理：裁剪 + 编码 + 删临时文件 + 返回结果
    # ========================================================================
    def _postprocess(self, task: MattingTask, raw_result: Dict, inference_time: float):
        """后处理：裁剪透明边缘 + 编码 PNG + 删除临时文件"""
        postprocess_start = time.time()
        try:
            output_img = raw_result[OutputKeys.OUTPUT_IMG]

            # 裁剪透明边缘
            if task.trim_edges:
                output_img = self._trim_transparent(output_img)

            # 编码为 PNG bytes
            _, buffer = cv2.imencode('.png', output_img)
            output_bytes = buffer.tobytes()

            result = {
                'success': True,
                'data': output_bytes,
                'format': 'png'
            }

        except Exception as e:
            result = {'success': False, 'error': f'后处理失败: {e}'}

        finally:
            # 删除临时文件
            if task.temp_path:
                try:
                    os.remove(task.temp_path)
                except OSError:
                    pass

        # 线程安全地设置 future 结果
        if task.loop and not task.future.done():
            task.loop.call_soon_threadsafe(task.future.set_result, result)

        # 更新统计
        postprocess_time = time.time() - postprocess_start
        self._update_stats(inference_time, postprocess_time)

    # ========================================================================
    # 工具方法
    # ========================================================================
    def _trim_transparent(self, img_array: np.ndarray) -> np.ndarray:
        """裁剪透明边缘"""
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
            bbox = pil_img.getbbox()
            if bbox:
                pil_img = pil_img.crop(bbox)
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        return img_array

    def _update_stats(self, inference_time: float, postprocess_time: float):
        """更新统计信息"""
        with self._stats_lock:
            n = self._stats['total_processed'] + 1
            self._stats['total_processed'] = n
            self._stats['avg_inference_time'] = (
                (self._stats['avg_inference_time'] * (n - 1) + inference_time) / n
            )
            total_time = inference_time + postprocess_time
            self._stats['avg_total_time'] = (
                (self._stats['avg_total_time'] * (n - 1) + total_time) / n
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
        stats['queue_size'] = self._queue.qsize()
        stats['ready_queue_size'] = self._ready_queue.qsize()
        stats['inference_workers'] = self.num_inference_workers
        return stats

    def shutdown(self):
        """关闭处理器"""
        self._shutdown = True
        # 等待所有推理线程结束
        for t in self._inference_threads:
            t.join(timeout=5.0)
        # 关闭线程池
        self._preprocess_pool.shutdown(wait=False)
        self._postprocess_pool.shutdown(wait=False)
