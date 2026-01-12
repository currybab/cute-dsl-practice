import cuda.tile as ct
import cupy

@ct.kernel
def sum_dim1_kernel(input, output, dim0: int, dim2: int, reduce_dim: int, REDUCE_TILE: ct.Constant[int], DIM2_TILE: ct.Constant[int]):
    # dim0, reduce, dim2 -> dim0, 1, dim2
    bidx = ct.bid(0)  # dim0 index
    bidy = ct.bid(1)  # reduce tile index
    bidz = ct.bid(2)  # dim2 tile index

    tile = ct.load(input, index=(bidx, bidy, bidz), shape=(1, REDUCE_TILE, DIM2_TILE))
    result = ct.sum(tile, axis=(0, 1))

    dim0_idx = ct.zeros((DIM2_TILE,), dtype=ct.int32) + bidx
    dim1_idx = ct.zeros((DIM2_TILE,), dtype=ct.int32)  # always 0
    dim2_idx = bidz * DIM2_TILE + ct.arange(DIM2_TILE, dtype=ct.int32)
    ct.atomic_add(output, (dim0_idx, dim1_idx, dim2_idx), result)


# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input, output, shape are all float32 device tensors
def solution(input, dim: int, output, shape, ndim: int):
    dim0 = 1
    dim2 = 1
    dim1 = int(shape[dim]) # reduce target
    for i in range(ndim):
        if i < dim:
            dim0 *= int(shape[i])
        elif i > dim:
            dim2 *= int(shape[i])

    DIM2_TILE = min(256, dim2)
    REDUCE_TILE = 16
    grid = (dim0, ct.cdiv(dim1, REDUCE_TILE), ct.cdiv(dim2, DIM2_TILE))

    input_reshaped = input.reshape((dim0, dim1, dim2))
    output_flat = output.reshape((dim0, 1, dim2))
    ct.launch(
        cupy.cuda.get_current_stream(),
        grid,
        sum_dim1_kernel,
        (input_reshaped, output_flat, dim0, dim2, dim1, REDUCE_TILE, DIM2_TILE)
    )

if __name__ == "__main__":
    import torch

    test_configs = [
        # (shape, dim)
        ((16, 128, 256), 1),
        ((32, 512, 512), 0),
        ((8, 1024, 1024), 2),
        ((64, 128, 128, 128), 2),
        ((4, 256, 256, 256), 1),
        ((128, 64, 64, 64), 3),
    ]

    print("Testing Sum Over Dimension:")
    all_passed = True

    for shape, reduce_dim in test_configs:
        output_shape = list(shape)
        output_shape[reduce_dim] = 1

        # Create random input tensor
        input_torch = torch.randn(*shape, dtype=torch.float32, device="cuda")
        output_torch = torch.zeros(*output_shape, dtype=torch.float32, device="cuda")

        # Convert to cupy
        input_cupy = cupy.asarray(input_torch)
        output_cupy = cupy.asarray(output_torch)
        shape_cupy = cupy.array(input_torch.shape, dtype=cupy.int32)

        # Run cuda.tile sum reduction
        solution(input_cupy, reduce_dim, output_cupy, shape_cupy, input_torch.ndim)

        # PyTorch reference
        expected = torch.sum(input_torch, dim=reduce_dim, keepdim=True)

        # Convert result back to torch for comparison
        result = torch.as_tensor(output_cupy, device="cuda")

        # Check correctness
        if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
            print(f"  ✓ shape={shape}, dim={reduce_dim}")
        else:
            diff = torch.abs(result - expected).max().item()
            mean_diff = torch.abs(result - expected).mean().item()
            print(f"  ✗ shape={shape}, dim={reduce_dim} - Max diff: {diff}, Mean diff: {mean_diff}")
            all_passed = False

    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
