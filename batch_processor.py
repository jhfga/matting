"""
批量抠图处理器 - 支持动态批处理和并发控制
"""
import asyncio
import io
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable
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
    trim_edges: bool = True  # 是否裁剪透明边缘


class DynamicBatcher:
    """
    动态批处理器
    - 收集多个请求并批量处理
    - 支持超时机制和最大批次数限制
    - 自动调整批次大小以优化吞吐量
    """
    
    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        max_wait_time: float = 0.05,
        max_queue_size: int = 100,
        num_workers: int = 2
    ):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        
        # 任务队列
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._shutdown = False
        
        # 模型和线程池
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._model_lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            'total_processed': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0
        }
        self._stats_lock = threading.Lock()
        
    def initialize(self):
        """初始化模型（在主线程中调用）"""
        print(f"[Batcher] 正在加载模型: {self.model_path}")
        self._model = pipeline(Tasks.universal_matting, model=self.model_path)
        print("[Batcher] 模型加载完成")
        
        # 预热模型 - 执行一次推理让GPU准备好
        print("[Batcher] 预热模型...")
        try:
            import numpy as np
            dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
            dummy_path = os.path.join(tempfile.gettempdir(), "warmup.png")
            cv2.imwrite(dummy_path, dummy_img)
            self._model([dummy_path])
            os.remove(dummy_path)
            print("[Batcher] 模型预热完成")
        except Exception as e:
            print(f"[Batcher] 预热警告: {e}")
        
        # 启动批处理循环
        asyncio.create_task(self._batch_loop())
        
    async def submit(self, task: MattingTask) -> bool:
        """提交任务到队列"""
        try:
            self._queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            return False
    
    async def _batch_loop(self):
        """批处理主循环 - 优化版：快速收集，立即处理"""
        while not self._shutdown:
            batch: List[MattingTask] = []
            
            # 优化：先快速收集，不等待，马上处理
            # 这样可以让GPU保持忙碌状态
            try:
                # 第一个任务等待超时
                first_task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.max_wait_time
                )
                batch.append(first_task)
                
                # 然后快速收集队列中的其他任务（不等待新请求）
                while len(batch) < self.max_batch_size:
                    try:
                        task = self._queue.get_nowait()
                        batch.append(task)
                    except asyncio.QueueEmpty:
                        break
                        
            except asyncio.TimeoutError:
                # 长时间没有任务，短暂休眠避免CPU空转
                await asyncio.sleep(0.01)
                continue
            
            if batch:
                # 立即处理批次
                asyncio.create_task(self._process_batch(batch))
    
    async def _process_batch(self, batch: List[MattingTask]):
        """处理一批任务"""
        start_time = time.time()
        
        try:
            # 在线程池中执行模型推理
            results = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._inference_batch,
                batch
            )
            
            # 设置每个任务的结果
            for task, result in zip(batch, results):
                if not task.future.done():
                    task.future.set_result(result)
                    
        except Exception as e:
            # 批次失败时，每个任务都设置异常
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(e)
        
        # 更新统计
        processing_time = time.time() - start_time
        with self._stats_lock:
            self._stats['total_processed'] += len(batch)
            self._stats['total_batches'] += 1
            n = self._stats['total_batches']
            self._stats['avg_batch_size'] = (
                (self._stats['avg_batch_size'] * (n - 1) + len(batch)) / n
            )
            self._stats['avg_processing_time'] = (
                (self._stats['avg_processing_time'] * (n - 1) + processing_time) / n
            )
        
        print(f"[Batcher] 处理批次: {len(batch)} 张, 耗时: {processing_time:.3f}s")
    
    def _inference_batch(self, batch: List[MattingTask]) -> List[Dict[str, Any]]:
        """
        执行批量推理（使用 ModelScope 原生批量支持）
        一次性把所有图片传给模型，让模型内部并行处理
        """
        results = [{} for _ in batch]  # 预初始化结果列表
        temp_paths = []
        
        try:
            with self._model_lock:
                # 第1步：并行保存所有图片到临时文件
                for i, task in enumerate(batch):
                    try:
                        img = Image.open(io.BytesIO(task.image_data)).convert('RGB')
                        temp_path = os.path.join(tempfile.gettempdir(), f"matting_{task.task_id}_{int(time.time()*1000)}.png")
                        img.save(temp_path)  # 使用默认压缩，保证图片质量
                        temp_paths.append(temp_path)
                    except Exception as e:
                        # 单张保存失败，记录错误并继续
                        temp_paths.append(None)
                        results[i] = {
                            'success': False,
                            'error': f'保存临时文件失败: {e}'
                        }
                
                # 第2步：过滤掉保存失败的，提取有效路径
                valid_paths = [p for p in temp_paths if p is not None]
                
                if valid_paths:
                    # 第3步：一次性传给模型进行批量推理！
                    # ModelScope pipeline 支持 list 输入，内部会并行处理
                    batch_results = self._model(valid_paths)
                    
                    # 第4步：整理结果
                    result_idx = 0
                    for i, task in enumerate(batch):
                        if temp_paths[i] is None:
                            # 之前保存失败的，继续下一个（result_idx 保持不变）
                            continue
                            
                        try:
                            # 获取对应结果
                            res = batch_results[result_idx]
                            output_img = res[OutputKeys.OUTPUT_IMG]
                            
                            # 根据参数决定是否裁剪透明边缘
                            if batch[i].trim_edges:
                                output_img = self._trim_transparent(output_img)
                            
                            # 编码为 bytes - 使用默认 PNG 压缩，保证图片质量
                            _, buffer = cv2.imencode('.png', output_img)
                            output_bytes = buffer.tobytes()
                            
                            results[i] = {
                                'success': True,
                                'data': output_bytes,
                                'format': 'png'
                            }
                            
                        except Exception as e:
                            results[i] = {
                                'success': False,
                                'error': f'处理失败: {e}'
                            }
                        result_idx += 1
                
        finally:
            # 清理所有临时文件 - 异步删除减少阻塞
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
        return results
    
    def _trim_transparent(self, img_array: np.ndarray) -> np.ndarray:
        """裁剪透明边缘"""
        # 转换为 PIL Image 处理透明边缘
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # 已有 Alpha 通道
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
            bbox = pil_img.getbbox()
            if bbox:
                pil_img = pil_img.crop(bbox)
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        return img_array
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            return self._stats.copy()
    
    def shutdown(self):
        """关闭处理器"""
        self._shutdown = True
        self._executor.shutdown(wait=True)
