import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

@cute.kernel
def gemm_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    M = cute.size(A, mode=[0])
    K = cute.size(A, mode=[1])
    N = cute.size(B, mode=[1])
    
    i = bidx * bdimx + tidx
    j = bidy * bdimy + tidy
    
    acc = 0.0
    if i < M and j < N:
        for k in range(K):
            acc += A[(i, k)] * B[(k, j)]
        C[(i, j)] = acc

@cute.jit
def simple_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    M = cute.size(A, mode=[0])
    K = cute.size(A, mode=[1])
    N = cute.size(B, mode=[1])
    block = (32, 32)
    grid = cute.ceil_div((M, N), block)
    cute.printf(grid)

    gemm_kernel(A, B, C).launch(
        grid=grid,
        block=block
    )

def test_conv1d(M, N, K):
    print(f"Testing M={M}, N={N}, K={K}")
    
    a_torch = torch.randn(M, K, dtype=torch.float32, device="cuda")
    b_torch = torch.randn(K, N, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    # Compile and run
    gemm_compiled = cute.compile(simple_gemm, from_dlpack(a_torch), from_dlpack(b_torch), from_dlpack(c_torch))
    gemm_compiled(from_dlpack(a_torch), from_dlpack(b_torch), from_dlpack(c_torch))
    
    # Verification
    expected = a_torch @ b_torch 
    
    torch.cuda.synchronize()
    is_correct = torch.allclose(c_torch, expected, atol=1e-4)
    print(f"  Verification: {'Success' if is_correct else 'Failure'}")
    if not is_correct:
        print(f"  Max diff: {(c_torch - expected).abs().max()}")
    return is_correct


if __name__ == "__main__":
    test_conv1d(1024, 1024, 1024)
