import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# Tile sizes
BM = cutlass.const_expr(128)
BN = cutlass.const_expr(128)
BK = cutlass.const_expr(32)

TM = cutlass.const_expr(4) 
TN = cutlass.const_expr(4)

@cute.kernel
def gemm_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, 
    tvA: cute.Layout, tvB: cute.Layout, tvC: cute.Layout,
    shapeA: cute.Shape, shapeB: cute.Shape, shapeC: cute.Shape,
    cA: cute.Tensor, cB: cute.Tensor, cC: cute.Tensor
):
    
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    tid = tidx + tidy * bdimx

    # Shared memory
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BM, BK)))
    sB = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BK, BN)))

    # Register C fragment
    rC = cute.make_rmem_tensor(tvC[1], cutlass.Float32)
    rC.fill(0.0)

    # C tile
    blkC = gC[None, (bidx, bidy)]
    thrC = cute.composition(blkC, tvC)[tid, None]

    # Predicate for C store
    blkC_c = cC[None, (bidx, bidy)]
    thrC_c = cute.composition(blkC_c, tvC)[tid, None]

    predC = cute.make_rmem_tensor(thrC.shape, cutlass.Boolean)
    for i in range(cute.size(predC)):
        predC[i] = cute.elem_less(thrC_c[i], shapeC)

    _, num_k_tiles = gA.shape[1]
    m_base = tidx * TM
    n_base = tidy * TN

    rB = cute.make_rmem_tensor(cute.make_layout((TN,)), cutlass.Float32)

    for bidk in range(num_k_tiles):
        blkA = gA[None, (bidx, bidk)]
        blkB = gB[None, (bidk, bidy)]

        # tile A, B
        thrA_g = cute.composition(blkA, tvA)[(tidx, tidy), None]
        thrB_g = cute.composition(blkB, tvB)[(tidx, tidy), None]

        # smem A, B
        thrA_s = cute.composition(sA, tvA)[tid, None]
        thrB_s = cute.composition(sB, tvB)[tid, None]

        # predication
        blkA_c = cA[None, (bidx, bidk)]
        blkB_c = cB[None, (bidk, bidy)]
        thrA_c = cute.composition(blkA_c, tvA)[tid, None]
        thrB_c = cute.composition(blkB_c, tvB)[tid, None]

        predA = cute.make_rmem_tensor(thrA_g.shape, cutlass.Boolean)
        predB = cute.make_rmem_tensor(thrB_g.shape, cutlass.Boolean)
        for i in range(cute.size(predA)):
            predA[i] = cute.elem_less(thrA_c[i], shapeA)
        for i in range(cute.size(predB)):
            predB[i] = cute.elem_less(thrB_c[i], shapeB)


        # gmem -> smem
        for i in range(cute.size(thrA_s)):
            thrA_s[i] = cutlass.Float32(0.0)
        for i in range(cute.size(thrB_s)):
            thrB_s[i] = cutlass.Float32(0.0)

        cute.basic_copy_if(predA, thrA_g, thrA_s)
        cute.basic_copy_if(predB, thrB_g, thrB_s)
        cute.arch.sync_threads()

        # ---- Clean + unrolled 4x4 outer-product accumulate ----
        # rC는 row-major(valC stride=(4,1))라서 linear idx = i*4 + j 사용
        for kk in cutlass.range_constexpr(BK):
            # 1) load B(kk, n_base:n_base+4) once
            for j in cutlass.range_constexpr(TN):
                rB[j] = sB[kk, n_base + j]

            # 2) for each i, load A(m_base+i, kk) once and accumulate 4 cols
            for i in cutlass.range_constexpr(TM):
                ai = sA[m_base + i, kk]
                row = i * TN
                for j in cutlass.range_constexpr(TN):
                    rC[row + j] += ai * rB[j]

        cute.arch.sync_threads()
    
    # Store C
    cute.basic_copy_if(predC, rC, thrC)

@cute.jit
def simple_tile_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Thread layout: 32x32 = 1024 threads
    thr_layout = cute.make_layout((32, 32), stride=(1, 32))

    valA = cute.make_layout((4, 1))
    valB = cute.make_layout((1, 4))

    # 핵심: C의 per-thread 4x4는 row-major
    valC = cute.make_layout((4, 4), stride=(4, 1))

    tilerA, tvA = cute.make_layout_tv(thr_layout, valA)
    tilerB, tvB = cute.make_layout_tv(thr_layout, valB)
    tilerC, tvC = cute.make_layout_tv(thr_layout, valC)

    gA = cute.zipped_divide(A, tilerA)  # ((TileM,TileK),(RestM,RestK))
    gB = cute.zipped_divide(B, tilerB)  # ((TileK,TileN),(RestK,RestN))
    gC = cute.zipped_divide(C, tilerC)  # ((TileM,TileN),(RestM,RestN))
    
    # predication
    shapeA = A.shape  # (M, K)
    shapeB = B.shape  # (K, N)
    shapeC = C.shape  # (M, N)
    
    cA = cute.zipped_divide(cute.make_identity_tensor(shapeA), tilerA)
    cB = cute.zipped_divide(cute.make_identity_tensor(shapeB), tilerB)
    cC = cute.zipped_divide(cute.make_identity_tensor(shapeC), tilerC)

    gemm_kernel(gA, gB, gC, tvA, tvB, tvC, shapeA, shapeB, shapeC, cA, cB, cC).launch(
        grid=gC.shape[1],
        block=thr_layout.shape,
    )


def test_gemm(M, N, K):
    print(f"Testing M={M}, N={N}, K={K}")

    a_torch = torch.randn(M, K, dtype=torch.float32, device="cuda")
    b_torch = torch.randn(K, N, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    gemm_compiled = cute.compile(
        simple_tile_gemm,
        from_dlpack(a_torch),
        from_dlpack(b_torch),
        from_dlpack(c_torch),
    )
    gemm_compiled(from_dlpack(a_torch), from_dlpack(b_torch), from_dlpack(c_torch))

    torch.cuda.synchronize()
    expected = a_torch @ b_torch
    ok = torch.allclose(c_torch, expected, atol=1e-3, rtol=1e-3)
    print(f"  Verification: {'Success' if ok else 'Failure'}")
    if not ok:
        print(f"  Max diff: {(c_torch - expected).abs().max()}")
    return ok


if __name__ == "__main__":
    test_gemm(512, 1024, 512)
    test_gemm(333, 333, 333)
