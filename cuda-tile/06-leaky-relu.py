import cuda.tile as ct
import cupy

@ct.kernel
def leaky_relu_kernel(input, alpha: float,output, n: int, m: int, n_tile: ct.Constant[int], m_tile: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    input_tile = ct.load(input, index=(bidx, bidy), shape=(n_tile, m_tile))
    # output_tile = mask * input_tile + (1 - mask) * alpha * input_tile
    output_tile = ct.where(input_tile > 0, input_tile, alpha * input_tile)
    ct.store(output, index=(bidx, bidy), tile=output_tile)
    
# Input
# - Matrix A of size (M, N)
# - α value (slope for negative values)
# Output
# - Matrix C of size (M, N)
#
# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input, output are all float32 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    n_tile = 32
    m_tile = 64
    grid = (ct.cdiv(n, n_tile), ct.cdiv(m, m_tile))
    ct.launch(cupy.cuda.get_current_stream(), grid, leaky_relu_kernel, (input, alpha, output, n, m, n_tile, m_tile))


if __name__ == "__main__":
    import torch

    # Test with 6144x4096 tensor
    n, m = 6144, 4096
    alpha = 0.1

    # Create random input tensor (with negative values for ReLU testing)
    input_torch = torch.randn(n, m, dtype=torch.float32, device="cuda")
    output_torch = torch.zeros(n, m, dtype=torch.float32, device="cuda")

    # Convert to cupy for cuda.tile
    input_cupy = cupy.asarray(input_torch)
    output_cupy = cupy.asarray(output_torch)

    # Run cuda.tile Leaky ReLU
    solution(input_cupy, alpha, output_cupy, n, m)

    # PyTorch reference
    expected = torch.nn.functional.leaky_relu(input_torch, negative_slope=alpha)

    # Convert result back to torch for comparison
    result = torch.as_tensor(output_cupy, device="cuda")

    # Check correctness
    if torch.allclose(result, expected):
        print(f"✓ ReLU test passed! Shape: ({n}, {m})")
    else:
        diff = torch.abs(result - expected).max().item()
        print(f"✗ ReLU test failed! Max diff: {diff}")
