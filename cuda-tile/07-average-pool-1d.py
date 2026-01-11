import cuda.tile as ct
import cupy

@ct.kernel
def average_pool_1d_kernel(input, kernel_size: int, stride: int, padding: int, output, H: int, TILE_SIZE: ct.Constant[int]):
    bidx = ct.bid(0)

    out_indices = bidx * TILE_SIZE + ct.arange(TILE_SIZE, dtype=ct.int32)
    init_indices = out_indices * stride - padding
    acc = ct.zeros((TILE_SIZE,), dtype=ct.float32)
    for k in range(kernel_size):
        acc = acc + ct.gather(input, init_indices + k)
    acc = acc / kernel_size
    ct.store(output, index=(bidx,), tile=acc)

# output[i]= 1/k * sum(input[S*i+m−P]) m=0~k-1
#
# Input:
# - Matrix input of size H (input tensor)
# - kernel_size (k): Size of the pooling window
# - stride (S): Step size between window positions
# - padding (P): Number of zero-padding elements added on all sides
# Output:
# - Matrix output of size ceil((H + 2P - k) / S + 1) (output tensor)
# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input, output are all float32 device tensors
def solution(input, kernel_size: int, stride: int, padding: int, output, H: int):
    out_size = (H + 2 * padding - kernel_size) // stride + 1
    TILE_SIZE = 256
    grid = (ct.cdiv(out_size, TILE_SIZE),)
    ct.launch(cupy.cuda.get_current_stream(), grid, average_pool_1d_kernel, (input, kernel_size, stride, padding, output, H, TILE_SIZE))


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Test parameters from the provided example
    batch_size = 64
    in_channels = 128
    input_length = 65536
    kernel_size = 8
    stride = 1
    padding = 4

    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    print(f"Testing 1D Average Pooling:")
    print(f"  Input shape: ({batch_size}, {in_channels}, {input_length})")
    print(f"  Output shape: ({batch_size}, {in_channels}, {output_length})")
    print(f"  kernel_size={kernel_size}, stride={stride}, padding={padding}")

    # Create random input tensor
    input_torch = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    output_torch = torch.zeros(batch_size, in_channels, output_length, dtype=torch.float32, device="cuda")

    # PyTorch reference
    avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    expected = avg_pool(input_torch)

    # Run cuda.tile Average Pooling for each batch and channel
    for b in range(batch_size):
        for c in range(in_channels):
            input_cupy = cupy.asarray(input_torch[b, c])
            output_cupy = cupy.asarray(output_torch[b, c])
            solution(input_cupy, kernel_size, stride, padding, output_cupy, input_length)
            output_torch[b, c] = torch.as_tensor(output_cupy, device="cuda")

    # Check correctness
    if torch.allclose(output_torch, expected, rtol=1e-4, atol=1e-5):
        print(f"✓ Average Pool 1D test passed!")
    else:
        diff = torch.abs(output_torch - expected).max().item()
        mean_diff = torch.abs(output_torch - expected).mean().item()
        print(f"✗ Average Pool 1D test failed!")
        print(f"  Max diff: {diff}")
        print(f"  Mean diff: {mean_diff}")
