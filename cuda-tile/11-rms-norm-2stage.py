import cuda.tile as ct
import cupy

EPSILON = 1e-5

# Kernel 1: compute rstd (reduction along N)
@ct.kernel
def compute_rstd_kernel(X, Rstd, N: int, B_TILE: ct.Constant[int], N_TILE: ct.Constant[int]):
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, N_TILE))
    sum_sq = ct.zeros((B_TILE,), dtype=ct.float32)

    for ni in range(num_tiles):
        X_tile = ct.load(X, index=(bid, ni), shape=(B_TILE, N_TILE))
        sum_sq += ct.sum(X_tile * X_tile, axis=1)

    rstd = 1 / ct.sqrt(sum_sq / N + EPSILON)
    ct.store(Rstd, index=(bid,), tile=rstd)


# Kernel 2: normalize (elementwise, fully parallel)
@ct.kernel
def normalize_kernel(X, Rstd, Y, B_TILE: ct.Constant[int], N_TILE: ct.Constant[int]):
    bid_b = ct.bid(0)  # batch tile
    bid_n = ct.bid(1)  # N tile

    rstd = ct.load(Rstd, index=(bid_b,), shape=(B_TILE,))
    rstd = ct.expand_dims(rstd, 1)
    X_tile = ct.load(X, index=(bid_b, bid_n), shape=(B_TILE, N_TILE))
    Y_tile = X_tile * rstd
    ct.store(Y, index=(bid_b, bid_n), tile=Y_tile)


def solution(X, Y, B: int, N: int):
    B_TILE = 32
    N_TILE = 128
    stream = cupy.cuda.get_current_stream()

    # Allocate intermediate buffer for rstd
    Rstd = cupy.zeros((B,), dtype=cupy.float32)

    # Kernel 1: compute rstd
    grid1 = (ct.cdiv(B, B_TILE),)
    ct.launch(stream, grid1, compute_rstd_kernel, (X, Rstd, N, B_TILE, N_TILE))

    # Kernel 2: normalize (parallel over B and N)
    grid2 = (ct.cdiv(B, B_TILE), ct.cdiv(N, N_TILE))
    ct.launch(stream, grid2, normalize_kernel, (X, Rstd, Y, B_TILE, N_TILE))


if __name__ == "__main__":
    import torch

    class RMSNormRef:
        def __init__(self, epsilon=1e-5):
            self.epsilon = epsilon

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
                rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
                return x / rms

    test_configs = [
        (16, 128),
        (32, 256),
        (64, 512),
        (128, 1024),
        (256, 2048),
    ]

    print("Testing RMS Normalization:")
    all_passed = True
    ref = RMSNormRef(epsilon=EPSILON)

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
