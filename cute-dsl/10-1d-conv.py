import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import argparse
import sys

@cute.kernel
def conv1d_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor, gIdx: cute.Tensor, tv_layout: cute.Layout, stride: cutlass.Int32, padding: cutlass.Int32, total_out_size: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    # Partition output and index tensors
    # gY and gIdx are (Tile, Block)
    # Composition with tv_layout gives (Thread, Value) -> Tile
    tidfrgY = cute.composition(gY[(None, bidx)], tv_layout)
    tidfrgIdx = cute.composition(gIdx[(None, bidx)], tv_layout)
    
    # Thread-local views
    thrY = tidfrgY[(tidx, None)]
    thrIdx = tidfrgIdx[(tidx, None)]
    
    K = cute.size(gW)
    L = cute.size(gX)
    
    for i in range(cute.size(thrY)):
        # In CuTe, identity tensors for 1D shapes might return a 1-tuple (index,)
        # We extract the scalar to compare with total_out_size.
        out_idx_tuple = thrIdx[i]
        out_idx = out_idx_tuple[0]
        
        if out_idx < total_out_size:
            acc = 0.0
            start_idx = out_idx * stride - padding
            
            for k in range(K):
                input_idx = start_idx + k
                if input_idx >= 0 and input_idx < L:
                    acc += gX[input_idx] * gW[k]
            
            thrY[i] = acc

@cute.jit
def conv1d(X: cute.Tensor, W: cute.Tensor, Y: cute.Tensor, stride: cutlass.Int32, padding: cutlass.Int32):
    # Idiomatic CuTe: Define layouts and tiling
    threads_per_block = 128
    values_per_thread = 1
    
    thr_layout = cute.make_layout(threads_per_block)
    val_layout = cute.make_layout(values_per_thread)
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    # Create an identity tensor to track global indices through transformations
    # This is a core CuTe concept for coordinate mapping
    idx_tensor = cute.make_identity_tensor(Y.layout.shape)
    
    # Partition tensors into tiles (zipped_divide)
    gY = cute.zipped_divide(Y, tiler)
    gIdx = cute.zipped_divide(idx_tensor, tiler)
    
    # Pass total size explicitly to handle boundary conditions safely
    total_out_size = cutlass.Int32(cute.size(Y))
    
    conv1d_kernel(X, W, gY, gIdx, tv_layout, stride, padding, total_out_size).launch(
        grid=(cute.size(gY, mode=[1]), 1, 1),
        block=(threads_per_block, 1, 1)
    )

def test_conv1d(L, K, S, P):
    print(f"Testing L={L}, K={K}, S={S}, P={P}")
    
    # Calculate output size: (L + 2*P - K) // S + 1
    out_size = (L + 2 * P - K) // S + 1
    
    x_torch = torch.randn(L, dtype=torch.float32, device="cuda")
    w_torch = torch.randn(K, dtype=torch.float32, device="cuda")
    y_torch = torch.zeros(out_size, dtype=torch.float32, device="cuda")
    
    # Compile and run
    conv1d_compiled = cute.compile(conv1d, from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    conv1d_compiled(from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    
    # Verification
    x_4d = x_torch.view(1, 1, L)
    w_4d = w_torch.view(1, 1, K)
    expected = torch.nn.functional.conv1d(x_4d, w_4d, stride=S, padding=P).view(-1)
    
    torch.cuda.synchronize()
    is_correct = torch.allclose(y_torch, expected, atol=1e-5)
    print(f"  Output size: {out_size}")
    print(f"  Verification: {'Success' if is_correct else 'Failure'}")
    if not is_correct:
        print(f"  Max diff: {(y_torch - expected).abs().max()}")
    return is_correct

def benchmark_conv1d(L=2**20, K=15, S=3, P=1, iters=100):
    print(f"Benchmarking: L={L}, K={K}, S={S}, P={P}, iters={iters}")
    
    out_size = (L + 2 * P - K) // S + 1
    x_torch = torch.randn(L, dtype=torch.float32, device="cuda")
    w_torch = torch.randn(K, dtype=torch.float32, device="cuda")
    y_torch = torch.zeros(out_size, dtype=torch.float32, device="cuda")
    
    # Compile
    conv1d_compiled = cute.compile(conv1d, from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    
    # Warmup
    for _ in range(10):
        conv1d_compiled(from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    torch.cuda.synchronize()
    
    # Timing CuTe
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        conv1d_compiled(from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    end_event.record()
    torch.cuda.synchronize()
    
    ms_cute = start_event.elapsed_time(end_event) / iters
    
    # Timing PyTorch
    x_4d = x_torch.view(1, 1, L)
    w_4d = w_torch.view(1, 1, K)
    
    # Warmup PyTorch
    for _ in range(10):
        _ = torch.nn.functional.conv1d(x_4d, w_4d, stride=S, padding=P)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(iters):
        _ = torch.nn.functional.conv1d(x_4d, w_4d, stride=S, padding=P)
    end_event.record()
    torch.cuda.synchronize()
    
    ms_torch = start_event.elapsed_time(end_event) / iters
    
    print(f"  CuTe DSL: {ms_cute:.3f} ms")
    print(f"  PyTorch:  {ms_torch:.3f} ms")
    print(f"  Speedup:  {ms_torch / ms_cute:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_conv1d()
    else:
        test_cases = [
            (1024, 3, 1, 0),
            (1024, 3, 1, 1), # Same padding for K=3
            (1024, 5, 1, 2), # Same padding for K=5
            (1024, 3, 2, 1),
            (2048, 7, 3, 3),
            (512, 11, 4, 5),
            (100, 3, 1, 0),
        ]
        
        all_success = True
        for L, K, S, P in test_cases:
            if not test_conv1d(L, K, S, P):
                all_success = False
        
        print("\nOverall Status:", "PASS" if all_success else "FAIL")
