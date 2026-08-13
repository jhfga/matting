# 抠图 API 服务

基于 ModelScope universal-matting 模型的高性能抠图服务。

## 环境准备

### GPU 环境配置（推荐）

| TensorFlow 版本 | CUDA 版本 | cuDNN 版本 |
|----------------|-----------|------------|
| 2.10.0         | 11.2      | 8.1        |

#### 方法：使用 conda 安装
```bash
# 创建新环境（指定 Python 3.9）
conda create -n matting python=3.9
conda activate matting

# 安装 CUDA 工具包和 cuDNN（自动处理版本兼容）
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1

# 安装所有依赖
pip install -r requirements.txt
```

#### 验证 GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

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
