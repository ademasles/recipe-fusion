"""GPU sanity check and micro-benchmark: availability, matmul, AMP training throughput."""

import time

import torch
from torch import nn


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_matmul(n: int = 8192, repeats: int = 10) -> float:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    (a @ b).sum().item()  # warm-up
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    sync()
    return (time.perf_counter() - start) / repeats


def bench_transformer(batch: int = 64, tokens: int = 197, dim: int = 384, steps: int = 30) -> float:
    """Samples/s for forward+backward of a ViT-S-sized encoder in bf16 autocast."""
    layer = nn.TransformerEncoderLayer(dim, nhead=6, dim_feedforward=4 * dim, batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=12).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, tokens, dim, device="cuda")
    warmup = 5
    start = time.perf_counter()
    for step in range(steps + warmup):
        if step == warmup:  # restart the clock once warm-up steps are done
            sync()
            start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x).float().pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync()
    return batch * steps / (time.perf_counter() - start)


def main() -> None:
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible.")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} | {props.total_memory / 2**30:.1f} GiB")
    print(f"matmul 8192x8192 fp32: {bench_matmul() * 1e3:.1f} ms")
    print(f"ViT-S-sized encoder, bf16 AMP, fwd+bwd: {bench_transformer():.0f} samples/s")
    print(f"peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
