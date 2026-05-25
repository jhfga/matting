# 抠图 API 服务

基于 ModelScope universal-matting 模型的高性能抠图服务。

## 环境准备

### GPU 环境配置（推荐）

**注意**：TensorFlow 2.11+ 在原生 Windows 上不再支持 GPU，因此使用 **TensorFlow 2.10**（Windows 原生支持的最后一个 GPU 版本）。

| TensorFlow 版本 | CUDA 版本 | cuDNN 版本 |
|----------------|-----------|------------|
| 2.10.0         | 11.2      | 8.1        |

#### 方法1：使用 conda 安装（推荐）

**注意**：TensorFlow 2.10 需要 Python 3.7-3.10，请确保不要安装太新的 Python 版本。

```bash
# 创建新环境（指定 Python 3.9）
conda create -n matting python=3.9
conda activate matting

# 安装 CUDA 工具包和 cuDNN（自动处理版本兼容）
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1

# 安装 TensorFlow
pip install tensorflow==2.10.0

# 安装其他依赖（去掉 tensorflow）
pip install fastapi uvicorn python-multipart aiohttp Pillow opencv-python numpy modelscope pydantic
```

#### 方法2：系统级 CUDA 安装

如果你不想用 conda：
1. 下载并安装 CUDA 11.2：https://developer.nvidia.com/cuda-11.2.0-download-archive
2. 下载并安装 cuDNN 8.1：https://developer.nvidia.com/cudnn
3. 将 CUDA 和 cuDNN 的 bin 目录添加到系统 PATH

#### 验证 GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### CPU-only 环境

如果没有 GPU，可以修改 `requirements.txt`，将 `tensorflow==2.10.0` 替换为 `tensorflow-cpu==2.10.0`

## 启动服务

### 1. 下载模型

```bash
modelscope download --model iic/cv_unet_universal-matting --local_dir ./models/cv_unet_universal-matting
```

### 2. 模型路径说明

模型路径已预配置为 `./models/cv_unet_universal-matting`，与上一步下载目录一致。如需自定义路径，可通过环境变量设置：

```bash
set MATTING_MODEL_PATH=你的模型路径
```

### 3. 启动服务

```bash
python main.py
```

服务默认在 `0.0.0.0:8000` 启动，其他电脑可通过服务器IP访问。

## 调用 API

### 接口地址

```
POST http://<服务器IP>:8000/api/matting
```

### 请求参数

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "trim_edges": true,
  "output_format": "png"
}
```

| 参数 | 类型 | 说明 |
|------|------|------|
| image | string | Base64 编码的图片（带 data:image 前缀） |
| trim_edges | boolean | 是否裁剪透明边缘（默认true） |
| output_format | string | 输出格式：png, jpg, webp（默认png） |

### 响应格式

```json
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "format": "png",
  "processing_time": 0.523
}
```

### Python 调用示例

```python
import requests
import base64

# 服务器地址
API_URL = "http://192.168.1.4:8000"

# 读取图片并转base64
with open("input.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# 调用API
resp = requests.post(
    f"{API_URL}/api/matting",
    json={"image": f"data:image/png;base64,{img_base64}"},
    timeout=60
)

# 保存结果
result = resp.json()
if result["success"]:
    img_data = result["image"].split(",")[1]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(img_data))
```

## 其他接口

- **健康检查**: `GET /health`
- **API文档**: `http://<服务器IP>:8000/docs`
