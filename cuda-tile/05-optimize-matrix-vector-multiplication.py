import cuda.tile as ct
import cupy

@ct.kernel
def mat_vec_mul_kernel(A, B, C, M: int, K: int, M_TILE: ct.Constant[int], K_TILE: ct.Constant[int], NUM_K_TILES: ct.Constant[int]):
    bidx = ct.bid(0)

    acc = ct.zeros((M_TILE,), dtype=ct.float32)

    for k_block in range(NUM_K_TILES):
        a_tile = ct.load(A, index=(bidx, k_block), shape=(M_TILE, K_TILE), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k_block,), shape=(K_TILE,), padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.sum(a_tile * b_tile, axis=1)

    ct.store(C, index=(bidx,), tile=acc)
    

# Input
# - Matrix A of size (M, K)
# - Vector B of size K
# Output
# - Vector C of size M
# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input_a, input_b, output_c are all float32 device tensors
def solution(input_a, input_b, output_c, m: int, k: int):
    M_TILE = 64
    K_TILE = 512
    NUM_K_TILES = ct.cdiv(k, K_TILE)
    grid = (ct.cdiv(m, M_TILE),)
    ct.launch(cupy.cuda.get_current_stream(), grid, mat_vec_mul_kernel, (input_a, input_b, output_c, m, k, M_TILE, K_TILE, NUM_K_TILES))


if __name__ == "__main__":
    import torch

    # Test parameters
    M = 2048
    K = 131072

    # Create random input matrix and vector
    A_torch = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B_torch = torch.randn(K, dtype=torch.float32, device="cuda")
    C_torch = torch.zeros(M, dtype=torch.float32, device="cuda")

    # Convert to cupy for cuda.tile
    A_cupy = cupy.asarray(A_torch)
    B_cupy = cupy.asarray(B_torch)
    C_cupy = cupy.asarray(C_torch)

    # Run cuda.tile matrix-vector multiplication
    solution(A_cupy, B_cupy, C_cupy, M, K)

    # PyTorch reference: C = A @ B
    expected = torch.matmul(A_torch, B_torch)

    # Convert result back to torch for comparison
    result = torch.as_tensor(C_cupy, device="cuda")

    # Check correctness
    if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
        print(f"✓ Matrix-Vector Multiplication test passed! M={M}, K={K}")
    else:
        diff = torch.abs(result - expected).max().item()
        print(f"✗ Matrix-Vector Multiplication test failed! Max diff: {diff}")
