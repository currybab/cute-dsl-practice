import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def vector_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    blk_coord = (None, bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    # Simple load and store for max performance
    # CuTe will automatically vectorize this if the layout allows
    thrC.store(thrA.load() + thrB.load())


# Note: d_input1, d_input2, d_output are all device tensors
@cute.jit
def solution(d_input1: cute.Tensor, d_input2: cute.Tensor, d_output: cute.Tensor, n: cute.Int32, verbose: bool):
    # Optimizing for occupancy and vectorization
    thr_layout = cute.make_layout(128)
    val_layout = cute.make_layout(8)
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    if verbose:
        cute.printf("tv_layout: {}", tv_layout)
    
    gA = cute.zipped_divide(d_input1, tiler)
    gB = cute.zipped_divide(d_input2, tiler)
    gC = cute.zipped_divide(d_output, tiler)

    if verbose:
        cute.printf("gA layout: {}", gA.layout)

    vector_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=(cute.size(gC, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1)
    )


if __name__ == "__main__":
    N = 2**24 # 16M elements
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, dtype=torch.float32, device="cuda")
    C = torch.zeros(N, dtype=torch.float32, device="cuda")
    
    solution_compiled = cute.compile(solution, from_dlpack(A), from_dlpack(B), from_dlpack(C), cute.Int32(N), True)
    
    # Warmup (show output)
    print("--- Warmup (Verbose=True) ---")
    solution_compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), cute.Int32(N), True)
    torch.cuda.synchronize()

    # Timing CuTe (hide output)
    print("--- Timing CuTe (Verbose=False) ---")
    iters = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        solution_compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), cute.Int32(N), False)
    end_event.record()
    torch.cuda.synchronize()

    ms_cute = start_event.elapsed_time(end_event) / iters
    gbps_cute = (3 * N * 4) / (ms_cute * 1e6)
    print(f"CuTe: {ms_cute:.3f} ms, {gbps_cute:.2f} GB/s")

    # Timing PyTorch
    print("--- Timing PyTorch ---")
    # Warmup PyTorch
    _ = A + B
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        _ = A + B
    end_event.record()
    torch.cuda.synchronize()

    ms_torch = start_event.elapsed_time(end_event) / iters
    gbps_torch = (3 * N * 4) / (ms_torch * 1e6)
    print(f"PyTorch: {ms_torch:.3f} ms, {gbps_torch:.2f} GB/s")

    print(f"Speedup (CuTe/PyTorch): {ms_torch / ms_cute:.2f}x")
    
    C_torch = A + B
    print(f"torch.allclose(C_torch, C): {torch.allclose(C_torch, C)}")
