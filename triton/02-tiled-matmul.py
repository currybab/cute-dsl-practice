import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    # group programs along M to improve L2 locality
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group // grid_n)
    pid_n = pid_in_group % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n * stride_bn,
            mask=((k + offs_k)[:, None] < K) & (offs_n < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + offs_m * stride_cm + offs_n * stride_cn, acc,
             mask=(offs_m < M) & (offs_n < N))


def simple_matmul(A: tl.tensor, B: tl.tensor, C: tl.tensor):
    M, K = A.shape
    N = B.shape[1]
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_M = 4
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )

def test_conv1d(M, N, K):
    print(f"Testing M={M}, N={N}, K={K}")
    
    a_torch = torch.randn(M, K, dtype=torch.float32, device="cuda")
    b_torch = torch.randn(K, N, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    # Compile and run
    simple_matmul(a_torch, b_torch, c_torch)
    torch.cuda.synchronize()

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
