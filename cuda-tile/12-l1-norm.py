import cuda.tile as ct
import cupy

EPSILON = 1e-10

@ct.kernel
def l1_norm_kernel(X, Y, B: int, D: int, B_TILE: ct.Constant[int], D_TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    num_d_tiles = ct.num_tiles(X, axis=1, shape=(B_TILE, D_TILE))
    sum_l1 = ct.full((B_TILE, 1), EPSILON, ct.float32)

    for di in range(num_d_tiles):
        X_tile = ct.load(X, index=(bidx, di), shape=(B_TILE, D_TILE))
        sum_l1 += ct.sum(ct.where(X_tile > 0, X_tile, -X_tile), axis=1, keepdims=True)

    mean_l1 = sum_l1 / D  # sum → mean

    for di in range(num_d_tiles):
        X_tile = ct.load(X, index=(bidx, di), shape=(B_TILE, D_TILE))
        Y_tile = X_tile / mean_l1
        ct.store(Y, index=(bidx, di), tile=Y_tile)
        

# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: X, Y are all float32 device tensors
def solution(X, Y, B: int, D: int):
    B_TILE = 32
    D_TILE = 128
    grid = (ct.cdiv(B, B_TILE),)
    ct.launch(cupy.cuda.get_current_stream(), grid, l1_norm_kernel, (X, Y, B, D, B_TILE, D_TILE))


if __name__ == "__main__":
    import torch

    class L1NormRef:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
                return x / torch.mean(torch.abs(x), dim=1, keepdim=True)

    test_configs = [
        (16, 128),
        (32, 256),
        (64, 512),
        (128, 1024),
        (256, 2048),
    ]

    print("Testing L1 Normalization:")
    all_passed = True
    ref = L1NormRef()

    for B, N in test_configs:
        X_torch = torch.randn(B, N, dtype=torch.float32, device="cuda")
        Y_torch = torch.zeros(B, N, dtype=torch.float32, device="cuda")

        X_cupy = cupy.asarray(X_torch)
        Y_cupy = cupy.asarray(Y_torch)

        solution(X_cupy, Y_cupy, B, N)

        expected = ref(X_torch)
        result = torch.as_tensor(Y_cupy, device="cuda")

        if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
            print(f"  ✓ B={B}, N={N}")
        else:
            diff = torch.abs(result - expected).max().item()
            print(f"  ✗ B={B}, N={N} - Max diff: {diff}")
            all_passed = False

    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
