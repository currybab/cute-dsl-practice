import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def tv_vectorized_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    # Select the thread-block tile
    # 현재 GPU 블록이 담당하는 타일 크기의 작은 조각으로 만듬.
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # Compose TV layout to map (tid, vid) -> physical address
    # 더 이상 (m, n) 좌표가 아니라 (tid, vid) 좌표로 데이터를 바라보게 함.
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Slice per-thread vector
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC.store(thrA.load() + thrB.load())


@cute.jit
def tv_vectorized_add(x: cute.Tensor, y: cute.Tensor, dst: cute.Tensor):
    # Thread (4,32): 4 groups along M (row), 32 contiguous threads along N (col)
    # Value (4,8): each thread handles 4 rows x 8 contiguous values
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    cute.printf("tiler_mn: {}", tiler_mn)
    cute.printf("tv_layout: {}", tv_layout)

    gA = cute.zipped_divide(x, tiler_mn)
    gB = cute.zipped_divide(y, tiler_mn)
    gC = cute.zipped_divide(dst, tiler_mn)
    cute.printf("gC.shape: {}", gC.shape)
    cute.printf("gC.layout: {}", gC.layout)
    cute.printf("gC.size: {}", cute.size(gC, mode=[1]))
    cute.printf("tv_layout.size: {}", cute.size(tv_layout, mode=[0]))

    tv_vectorized_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1], # zipped_divide 좌측은 tile 크기, 나머지는 그리드(타일들의 배치).
        block=[cute.size(tv_layout, mode=[0]), 1, 1], # thread layout
    )

M, N = 2048, 2048
a = torch.randn(M, N, dtype=torch.float32, device="cuda")
b = torch.randn(M, N, dtype=torch.float32, device="cuda")
c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

vadd_compiled = cute.compile(tv_vectorized_add, from_dlpack(a), from_dlpack(b), from_dlpack(c))
vadd_compiled(from_dlpack(a), from_dlpack(b), from_dlpack(c))

c_torch = a + b
print(f"torch.allclose(c_torch, c): {torch.allclose(c_torch, c)}")
