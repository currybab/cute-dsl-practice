# This examples uses PyTorch which can be installed via `pip install torch`
# Make sure cuda toolkit 13.1+ is installed: https://developer.nvidia.com/cuda-downloads

import argparse

import cuda.tile as ct
import torch

TILE_SIZE = 1024

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result, tile_size: ct.Constant[int]):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(tile_size,))
    b_tile = ct.load(b, index=(block_id,), shape=(tile_size,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)


def launch_vector_add(a, b, result, tile_size=TILE_SIZE, stream=None):
    if stream is None:
        stream = torch.cuda.current_stream()
    grid = (ct.cdiv(a.shape[0], tile_size), 1, 1)
    ct.launch(stream, grid, vector_add_kernel, (a, b, result, tile_size))


def test_vector_add(vector_size, tile_size=TILE_SIZE, dtype=torch.float32):
    print(f"Testing N={vector_size}, tile_size={tile_size}, dtype={dtype}")

    a = torch.randn(vector_size, dtype=dtype, device="cuda")
    b = torch.randn(vector_size, dtype=dtype, device="cuda")
    result = torch.empty_like(a)

    launch_vector_add(a, b, result, tile_size=tile_size)
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(result, a + b)
    except AssertionError:
        print("  Verification: Failure")
        max_diff = (result - (a + b)).abs().max().item()
        print(f"  Max diff: {max_diff}")
        return False

    print("  Verification: Success")
    return True


def benchmark_vector_add(
    vector_size=2**24,
    tile_size=TILE_SIZE,
    dtype=torch.float32,
    iters=100,
    warmup=1,
):
    print(f"Benchmarking: N={vector_size}, tile_size={tile_size}, dtype={dtype}, iters={iters}")

    a = torch.randn(vector_size, dtype=dtype, device="cuda")
    b = torch.randn(vector_size, dtype=dtype, device="cuda")
    out_cutile = torch.empty_like(a)

    print("--- Warmup cuTile ---")
    for _ in range(warmup):
        launch_vector_add(a, b, out_cutile, tile_size=tile_size)
    torch.cuda.synchronize()

    print("--- Timing cuTile ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        launch_vector_add(a, b, out_cutile, tile_size=tile_size)
    end_event.record()
    torch.cuda.synchronize()

    ms_cutile = start_event.elapsed_time(end_event) / iters
    gbps_cutile = (3 * vector_size * a.element_size()) / (ms_cutile * 1e6)
    print(f"cuTile: {ms_cutile:.3f} ms, {gbps_cutile:.2f} GB/s")

    print("--- Warmup Torch ---")
    for _ in range(warmup):
        _ = a + b
    torch.cuda.synchronize()

    print("--- Timing Torch ---")
    start_event.record()
    for _ in range(iters):
        _ = a + b
    end_event.record()
    torch.cuda.synchronize()

    ms_torch = start_event.elapsed_time(end_event) / iters
    gbps_torch = (3 * vector_size * a.element_size()) / (ms_torch * 1e6)
    print(f"Torch:  {ms_torch:.3f} ms, {gbps_torch:.2f} GB/s")

    print(f"Speedup (cuTile/Torch): {ms_torch / ms_cutile:.2f}x")

    out_torch = a + b
    ok = torch.allclose(out_torch, out_cutile)
    print(f"torch.allclose(out_torch, out_cutile): {bool(ok)}")


def _dtype_from_str(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--n", type=int, default=2**24)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    args = parser.parse_args()

    dtype = _dtype_from_str(args.dtype)

    if args.benchmark:
        benchmark_vector_add(
            vector_size=args.n,
            tile_size=args.tile_size,
            dtype=dtype,
            iters=args.iters,
            warmup=args.warmup,
        )
    else:
        ok = test_vector_add(args.n, tile_size=args.tile_size, dtype=dtype)
        print("\nOverall Status:", "PASS" if ok else "FAIL")
