"""
抠图 API 压力测试脚本

模拟真实场景：在一段时间内，大量用户随机发起请求，统计每个用户的等待时间。

用法:
    python stress_test.py --image test.png --duration 30 --rate 10 --concurrency 20
"""

import argparse
import asyncio
import base64
import random
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

import aiohttp


# ============================================================================
# 统计
# ============================================================================
@dataclass
class Stats:
    total: int = 0
    success: int = 0
    fail: int = 0
    times: list = field(default_factory=list)        # 用户总等待时间
    queue_times: list = field(default_factory=list)   # 服务端排队时间
    errors: list = field(default_factory=list)
    sample_image: str = field(default=None)           # 保存一份结果图片 base64
    sample_format: str = field(default="png")

    # 实时采样（滑动窗口，用于运行时展示）
    recent_times: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def rate(self) -> float:
        return self.success / self.total if self.total else 0

    def percentile(self, data: list, p: float) -> float:
        if not data:
            return 0
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def avg(self, data: list) -> float:
        return sum(data) / len(data) if data else 0

    def summary(self, elapsed: float) -> str:
        t = self.times
        qt = self.queue_times
        lines = [
            f"{'=' * 55}",
            " 抠图 API 压力测试结果",
            f"{'=' * 55}",
            f" 测试窗口:      {elapsed:.1f}s",
            f" 请求总数:      {self.total}",
            f" 成功:          {self.success}",
            f" 失败:          {self.fail}",
            f" 成功率:        {self.rate * 100:.1f}%",
            f" 实际吞吐:      {self.total / elapsed:.1f} req/s",
            f"{'=' * 55}",
            " 用户等待时间 (请求发起 → 收到结果):",
            f"  平均:  {self.avg(t):.3f}s",
            f"  最小:  {min(t):.3f}s" if t else "  最小:  -",
            f"  最大:  {max(t):.3f}s" if t else "  最大:  -",
            f"  P50:   {self.percentile(t, 50):.3f}s",
            f"  P90:   {self.percentile(t, 90):.3f}s",
            f"  P95:   {self.percentile(t, 95):.3f}s",
            f"  P99:   {self.percentile(t, 99):.3f}s",
            f"{'=' * 55}",
            " 服务端处理 (不含排队):",
            f"  平均:  {self.avg([t[i] - qt[i] for i in range(len(t))]) if t else 0:.3f}s",
            " 服务端排队:",
            f"  平均:  {self.avg(qt):.3f}s",
            f"{'=' * 55}",
        ]
        if self.errors:
            lines.append(" 错误详情 (前5条):")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
        return "\n".join(lines)


# ============================================================================
# 工具
# ============================================================================
def encode_image(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    data = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ============================================================================
# 单个请求
# ============================================================================
async def do_request(
    session: aiohttp.ClientSession,
    url: str,
    image_b64: str,
    sem: asyncio.Semaphore,
    stats: Stats,
    trim_edges: bool,
    output_format: str,
    request_id: int,
):
    """单个用户请求：信号量控制并发上限"""
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{url}/api/matting",
                json={
                    "image": image_b64,
                    "trim_edges": trim_edges,
                    "output_format": output_format,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                t1 = time.perf_counter()
                data = await resp.json()

                if data.get("success"):
                    stats.success += 1
                    elapsed = t1 - t0
                    stats.times.append(elapsed)
                    stats.recent_times.append(elapsed)
                    qt = data.get("queue_time") or 0
                    stats.queue_times.append(qt)
                    # 保存第一份结果图片
                    if stats.sample_image is None and data.get("image"):
                        stats.sample_image = data["image"]
                        stats.sample_format = output_format
                else:
                    stats.fail += 1
                    stats.errors.append(data.get("error", "unknown"))
        except Exception as e:
            stats.fail += 1
            stats.errors.append(str(e))

        stats.total += 1


# ============================================================================
# 实时进度展示
# ============================================================================
async def progress_reporter(stats: Stats, stop_event: asyncio.Event):
    """每秒打印一次当前状态"""
    last_total = 0
    while not stop_event.is_set():
        await asyncio.sleep(1)
        recent = list(stats.recent_times)
        avg_recent = sum(recent) / len(recent) if recent else 0
        rps = stats.total - last_total
        last_total = stats.total
        print(
            f"  [运行中] 已完成: {stats.total} | "
            f"成功: {stats.success} | 失败: {stats.fail} | "
            f"近1s吞吐: {rps} req/s | "
            f"近50次平均等待: {avg_recent:.3f}s"
        )


# ============================================================================
# 主流程：模拟随机到达
# ============================================================================
async def run_random_arrival(
    url: str,
    image_b64: str,
    duration: float,
    rate: float,
    concurrency: int,
    trim_edges: bool,
    output_format: str,
):
    """
    在 duration 秒内，按照泊松过程随机发起请求，总请求数 ≈ rate * duration。
    每个请求到达时间由指数分布（均值 = 1/rate）的间隔确定。
    """
    stats = Stats()
    sem = asyncio.Semaphore(concurrency)

    total_expected = int(rate * duration)

    print(f"\n{'=' * 55}")
    print(f" 压力测试 - 随机到达模式")
    print(f"{'=' * 55}")
    print(f" 目标:        {url}/api/matting")
    print(f" 测试时长:    {duration}s")
    print(f" 目标速率:    {rate} req/s")
    print(f" 预计请求数:  ~{total_expected}")
    print(f" 最大并发:    {concurrency}")
    print(f"{'=' * 55}\n")

    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession(connector=connector) as session:
        # 启动进度展示协程
        reporter_task = asyncio.create_task(progress_reporter(stats, stop_event))

        t_start = time.perf_counter()
        request_id = 0

        # 使用指数分布生成随机到达间隔
        mean_interval = 1.0 / rate
        tasks_pending = []

        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration:
                break

            # 指数分布随机间隔
            delay = random.expovariate(1.0 / mean_interval)
            await asyncio.sleep(delay)

            # 再次检查是否超时
            if time.perf_counter() - t_start >= duration:
                break

            request_id += 1
            task = asyncio.create_task(
                do_request(session, url, image_b64, sem, stats, trim_edges, output_format, request_id)
            )
            tasks_pending.append(task)

        # 等待所有已发出的请求完成
        if tasks_pending:
            await asyncio.gather(*tasks_pending)

        t_end = time.perf_counter()

        # 停止进度展示
        stop_event.set()
        await reporter_task

    elapsed = t_end - t_start
    print(f"\n{stats.summary(elapsed)}")

    # 保存结果图片
    if stats.sample_image:
        fmt = stats.sample_format
        img_data = stats.sample_image.split(",", 1)[1] if "," in stats.sample_image else stats.sample_image
        out_path = f"stress_test_result.{fmt}"
        Path(out_path).write_bytes(base64.b64decode(img_data))
        print(f" 结果图片已保存: {out_path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="抠图 API 压力测试 - 模拟真实用户随机访问",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 30秒内，平均每秒10个用户随机访问，最大并发20
  python stress_test.py --image test.png --duration 30 --rate 10 --concurrency 20

  # 模拟突发流量：60秒，平均每秒50请求，最大并发100
  python stress_test.py --image test.png --duration 60 --rate 50 --concurrency 100

  # 指定远程服务器
  python stress_test.py --image girl.png --url http://192.168.1.100:8000 --duration 30 --rate 5
        """,
    )
    parser.add_argument("--image", required=True, help="本地测试图片路径")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API 服务地址")
    parser.add_argument("--duration", type=float, default=30, help="测试持续时间（秒），默认 30")
    parser.add_argument("--rate", type=float, default=10, help="目标请求速率（req/s），默认 10")
    parser.add_argument("--concurrency", type=int, default=20, help="最大并发连接数，默认 20")
    parser.add_argument("--no-trim", action="store_true", help="不裁剪透明边缘")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "webp"], help="输出格式")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片不存在: {args.image}")
        sys.exit(1)

    print(f"编码图片: {args.image}")
    image_b64 = encode_image(str(image_path))
    print(f"图片大小: {len(image_b64) / 1024:.1f} KB (base64)")

    asyncio.run(run_random_arrival(
        url=args.url.rstrip("/"),
        image_b64=image_b64,
        duration=args.duration,
        rate=args.rate,
        concurrency=args.concurrency,
        trim_edges=not args.no_trim,
        output_format=args.format,
    ))


if __name__ == "__main__":
    main()