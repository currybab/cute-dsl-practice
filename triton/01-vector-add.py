import triton
import triton.language as tl
import torch
import argparse

TILE_SIZE = 1024

@triton.jit
def vadd_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                      # like blockIdx.x
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)

def vadd(a, b, c):
    grid = ((a.shape[0] + TILE_SIZE - 1) // TILE_SIZE,)               # number of programs
    vadd_kernel[grid](a, b, c, a.shape[0], BLOCK_SIZE=TILE_SIZE)


def test_vector_add(vector_size, tile_size=TILE_SIZE, dtype=torch.float32):
    print(f"Testing N={vector_size}, tile_size={tile_size}, dtype={dtype}")

    a = torch.randn(vector_size, dtype=dtype, device="cuda")
    b = torch.randn(vector_size, dtype=dtype, device="cuda")
    result = torch.empty_like(a)

    vadd(a, b, result)
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
    out_triton = torch.empty_like(a)

    print("--- Warmup triton ---")
    for _ in range(warmup):
        vadd(a, b, out_triton)
    torch.cuda.synchronize()

    print("--- Timing triton ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        vadd(a, b, out_triton)
    end_event.record()
    torch.cuda.synchronize()

    ms_triton = start_event.elapsed_time(end_event) / iters
    gbps_triton = (3 * vector_size * a.element_size()) / (ms_triton * 1e6)
    print(f"triton: {ms_triton:.3f} ms, {gbps_triton:.2f} GB/s")

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

    print(f"Speedup (triton/Torch): {ms_torch / ms_triton:.2f}x")

    out_torch = a + b
    ok = torch.allclose(out_torch, out_triton)
    print(f"torch.allclose(out_torch, out_triton): {bool(ok)}")


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
