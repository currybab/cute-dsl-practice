import cuda.tile as ct
import cupy

@ct.kernel
def relu_kernel(input, output, n: int, m: int, n_tile: ct.Constant[int], m_tile: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    input_tile = ct.load(input, index=(bidx, bidy), shape=(n_tile, m_tile), padding_mode=ct.PaddingMode.ZERO)
    output_tile = ct.maximum(input_tile, 0.0)
    ct.store(output, index=(bidx, bidy), tile=output_tile)
    

# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int):
    n_tile = 64
    m_tile = 128
    grid = (ct.cdiv(n, n_tile), ct.cdiv(m, m_tile))
    ct.launch(cupy.cuda.get_current_stream(), grid, relu_kernel, (input, output, n, m, n_tile, m_tile))


if __name__ == "__main__":
    import torch

    # Test with 6144x4096 tensor
    n, m = 6144, 4096

    # Create random input tensor (with negative values for ReLU testing)
    input_torch = torch.randn(n, m, dtype=torch.float32, device="cuda")
    output_torch = torch.zeros(n, m, dtype=torch.float32, device="cuda")

    # Convert to cupy for cuda.tile
    input_cupy = cupy.asarray(input_torch)
    output_cupy = cupy.asarray(output_torch)

    # Run cuda.tile ReLU
    solution(input_cupy, output_cupy, n, m)

    # PyTorch reference
    expected = torch.relu(input_torch)

    # Convert result back to torch for comparison
    result = torch.as_tensor(output_cupy, device="cuda")

    # Check correctness
    if torch.allclose(result, expected):
        print(f"✓ ReLU test passed! Shape: ({n}, {m})")
    else:
        diff = torch.abs(result - expected).max().item()
        print(f"✗ ReLU test failed! Max diff: {diff}")
