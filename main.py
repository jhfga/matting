"""
高性能抠图 API 服务
- FastAPI + Uvicorn 提供高并发支持
- 动态批处理优化吞吐量
- 支持 base64 图片上传和 URL 图片
"""
import os
import sys

# 获取应用根目录（兼容 PyInstaller 打包和源码运行）
def get_app_dir():
    """获取应用根目录，兼容 PyInstaller 打包"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，使用 exe 所在目录
        return os.path.dirname(sys.executable)
    else:
        # 源码运行，使用脚本所在目录
        return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_dir()

# 在导入 tensorflow 之前设置 CUDA 环境变量
cuda_path = os.path.join(APP_DIR, "cuda_bin")
if os.path.exists(cuda_path):
    if cuda_path not in os.environ["PATH"]:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    # 回退到原始 conda 环境路径（开发时使用）
    _cuda_conda = r"C:\ProgramData\anaconda3\envs\matting\Library\bin"
    if _cuda_conda not in os.environ["PATH"]:
        os.environ["PATH"] = _cuda_conda + os.pathsep + os.environ["PATH"]
import io
import base64
import uuid
import asyncio
import time
from typing import Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from batch_processor import DynamicBatcher, MattingTask


# ============================================================================
# 配置 - 高性能优化版
# ============================================================================
MODEL_PATH = os.getenv("MATTING_MODEL_PATH", os.path.join(APP_DIR, "models", "cv_unet_universal-matting"))
# 优化：增大批处理大小，GPU可处理更多
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16"))
# 优化：等待时间 0.2s，平衡响应速度和批处理效率
MAX_WAIT_TIME = float(os.getenv("MAX_WAIT_TIME", "0.2"))
# 优化：增大队列，应对突发流量
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "200"))
# 优化：增加工作线程，支持更多并发批次
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
API_KEY = os.getenv("API_KEY")  # 可选的 API 密钥验证

# GPU 内存优化设置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ============================================================================
# Pydantic 模型
# ============================================================================
class MattingRequest(BaseModel):
    """抠图请求"""
    image: Optional[str] = Field(None, description="Base64 编码的图片")
    image_url: Optional[str] = Field(None, description="图片 URL")
    trim_edges: bool = Field(True, description="是否裁剪透明边缘")
    output_format: str = Field("png", description="输出格式: png, jpg, webp")


class MattingResponse(BaseModel):
    """抠图响应"""
    success: bool
    image: Optional[str] = Field(None, description="Base64 编码的结果图片")
    format: str
    processing_time: float
    queue_time: Optional[float] = None
    error: Optional[str] = None


class BatchMattingRequest(BaseModel):
    """批量抠图请求"""
    images: List[str] = Field(..., description="Base64 编码的图片列表")
    trim_edges: bool = Field(True, description="是否裁剪透明边缘")
    output_format: str = Field("png", description="输出格式")


class BatchMattingResponse(BaseModel):
    """批量抠图响应"""
    success: bool
    results: List[dict]
    total_processing_time: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    queue_size: int
    stats: dict


# ============================================================================
# 全局变量
# ============================================================================
batcher: Optional[DynamicBatcher] = None


# ============================================================================
# 工具函数 - 获取本地 IP
# ============================================================================
def get_local_ip():
    """获取本机局域网 IP 地址"""
    import socket
    try:
        # 创建一个 UDP 连接来获取本机 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ============================================================================
# 生命周期管理
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global batcher
    
    # 获取本地 IP
    local_ip = get_local_ip()
    
    # 启动时初始化
    print("=" * 60)
    print("🚀 启动抠图 API 服务")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   最大批次: {MAX_BATCH_SIZE}")
    print(f"   最大等待: {MAX_WAIT_TIME}s")
    print(f"   队列容量: {MAX_QUEUE_SIZE}")
    print(f"   工作线程: {NUM_WORKERS}")
    print("-" * 60)
    print("📡 服务访问地址:")
    print(f"   本机访问: http://127.0.0.1:8000")
    print(f"   局域网访问: http://{local_ip}:8000")
    print(f"   API 文档: http://{local_ip}:8000/docs")
    print("=" * 60)
    
    batcher = DynamicBatcher(
        model_path=MODEL_PATH,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_time=MAX_WAIT_TIME,
        max_queue_size=MAX_QUEUE_SIZE,
        num_workers=NUM_WORKERS
    )
    batcher.initialize()
    
    yield
    
    # 关闭时清理
    print("\n🛑 关闭服务...")
    batcher.shutdown()


# ============================================================================
# 创建 FastAPI 应用
# ============================================================================
app = FastAPI(
    title="智能抠图 API",
    description="高性能批量抠图服务 - 支持动态批处理和并发处理",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# 工具函数
# ============================================================================
async def download_image(url: str) -> bytes:
    """异步下载图片"""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400, 
                    detail=f"无法下载图片: HTTP {response.status}"
                )
            return await response.read()


def decode_base64_image(data: str) -> bytes:
    """解码 base64 图片"""
    try:
        if data.startswith('data:image'):
            data = data.split(',')[1]
        return base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")


def encode_image_to_base64(data: bytes, format: str) -> str:
    """编码图片为 base64"""
    mime_type = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'webp': 'image/webp'
    }.get(format.lower(), 'image/png')
    
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def convert_image_format(image_bytes: bytes, target_format: str) -> bytes:
    """转换图片格式"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # 处理透明背景
    if target_format.lower() in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        img = background
    
    output = io.BytesIO()
    save_format = 'JPEG' if target_format.lower() in ['jpg', 'jpeg'] else target_format.upper()
    img.save(output, format=save_format)
    return output.getvalue()


# ============================================================================
# API 路由
# ============================================================================
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    return {
        "service": "智能抠图 API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    stats = batcher.get_stats() if batcher else {}
    queue_size = batcher._queue.qsize() if batcher else 0
    
    return HealthResponse(
        status="healthy",
        model_loaded=batcher is not None,
        queue_size=queue_size,
        stats=stats
    )


@app.post("/api/matting", response_model=MattingResponse)
async def matting(request: MattingRequest):
    """
    单张图片抠图
    
    - 支持 base64 图片或图片 URL
    - 自动裁剪透明边缘
    - 支持多种输出格式
    """
    start_time = time.time()
    
    # 获取图片数据
    if request.image:
        image_data = decode_base64_image(request.image)
    elif request.image_url:
        image_data = await download_image(request.image_url)
    else:
        raise HTTPException(status_code=400, detail="请提供 image 或 image_url")
    
    # 创建任务
    task_id = str(uuid.uuid4())[:8]
    future = asyncio.get_event_loop().create_future()
    
    task = MattingTask(
        task_id=task_id,
        image_data=image_data,
        image_format="png",
        future=future,
        created_at=time.time(),
        trim_edges=request.trim_edges
    )
    
    # 提交到批处理器
    if not await batcher.submit(task):
        raise HTTPException(status_code=503, detail="服务器繁忙，请稍后重试")
    
    # 等待结果
    try:
        result = await asyncio.wait_for(future, timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="处理超时")
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', '处理失败'))
    
    # 转换格式（如果需要）
    output_data = result['data']
    if request.output_format.lower() != 'png':
        output_data = convert_image_format(output_data, request.output_format)
    
    end_time = time.time()
    processing_time = end_time - start_time
    queue_time = start_time - task.created_at  # 提交到开始处理的时间差
    
    return MattingResponse(
        success=True,
        image=encode_image_to_base64(output_data, request.output_format),
        format=request.output_format,
        processing_time=round(processing_time, 3),
        queue_time=round(queue_time, 3) if queue_time > 0.001 else None
    )


@app.post("/api/matting/batch", response_model=BatchMattingResponse)
async def batch_matting(request: BatchMattingRequest):
    """
    批量图片抠图
    
    - 一次最多处理 32 张图片
    - 自动分批并行处理
    - 返回每张图片的处理结果
    """
    if len(request.images) > 32:
        raise HTTPException(status_code=400, detail="单次最多处理 32 张图片")
    
    if len(request.images) == 0:
        raise HTTPException(status_code=400, detail="图片列表不能为空")
    
    start_time = time.time()
    results = []
    
    # 并行提交所有任务
    tasks = []
    for i, img_data in enumerate(request.images):
        try:
            image_bytes = decode_base64_image(img_data)
            task_id = f"{i}_{uuid.uuid4().hex[:6]}"
            future = asyncio.get_event_loop().create_future()
            
            task = MattingTask(
                task_id=task_id,
                image_data=image_bytes,
                image_format="png",
                future=future,
                created_at=time.time(),
                trim_edges=request.trim_edges
            )
            
            if await batcher.submit(task):
                tasks.append((i, task, future))
            else:
                results.append({
                    "index": i,
                    "success": False,
                    "error": "服务器队列已满"
                })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    # 等待所有任务完成
    for i, task, future in tasks:
        try:
            result = await asyncio.wait_for(future, timeout=60.0)
            
            if result['success']:
                output_data = result['data']
                if request.output_format.lower() != 'png':
                    output_data = convert_image_format(output_data, request.output_format)
                
                results.append({
                    "index": i,
                    "success": True,
                    "image": encode_image_to_base64(output_data, request.output_format),
                    "format": request.output_format
                })
            else:
                results.append({
                    "index": i,
                    "success": False,
                    "error": result.get('error', '处理失败')
                })
        except asyncio.TimeoutError:
            results.append({
                "index": i,
                "success": False,
                "error": "处理超时"
            })
    
    # 按索引排序
    results.sort(key=lambda x: x['index'])
    
    return BatchMattingResponse(
        success=True,
        results=results,
        total_processing_time=round(time.time() - start_time, 3)
    )


@app.post("/api/matting/upload")
async def matting_upload(
    file: UploadFile = File(...),
    trim_edges: bool = True,
    output_format: str = "png"
):
    """
    通过文件上传抠图
    
    - 支持 multipart/form-data 上传
    - 适合客户端直接上传文件
    """
    start_time = time.time()
    
    # 读取上传的文件
    image_data = await file.read()
    
    # 验证图片
    try:
        Image.open(io.BytesIO(image_data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图片文件")
    
    # 创建任务
    task_id = str(uuid.uuid4())[:8]
    future = asyncio.get_event_loop().create_future()
    
    task = MattingTask(
        task_id=task_id,
        image_data=image_data,
        image_format="png",
        future=future,
        created_at=time.time(),
        trim_edges=trim_edges
    )
    
    if not await batcher.submit(task):
        raise HTTPException(status_code=503, detail="服务器繁忙")
    
    # 等待结果
    try:
        result = await asyncio.wait_for(future, timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="处理超时")
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', '处理失败'))
    
    # 转换格式
    output_data = result['data']
    if output_format.lower() != 'png':
        output_data = convert_image_format(output_data, output_format)
    
    # 返回 base64
    return {
        "success": True,
        "image": encode_image_to_base64(output_data, output_format),
        "format": output_format,
        "processing_time": round(time.time() - start_time, 3)
    }


@app.get("/api/stats")
async def get_stats():
    """获取服务统计信息"""
    if not batcher:
        return {"error": "服务未初始化"}
    
    stats = batcher.get_stats()
    return {
        "stats": stats,
        "queue_size": batcher._queue.qsize(),
        "config": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_wait_time": MAX_WAIT_TIME,
            "max_queue_size": MAX_QUEUE_SIZE,
            "num_workers": NUM_WORKERS
        }
    }


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=1,  # 使用单个进程，批处理器内部管理并发
        loop="asyncio",
        access_log=True
    )
