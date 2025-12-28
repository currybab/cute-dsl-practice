import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def vectorized_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    mi = idx // n
    ni = idx % n
    gC[(None, (mi, ni))].store(gA[(None, (mi, ni))].load() + gB[(None, (mi, ni))].load())

@cute.jit
def vectorized_add(x: cute.Tensor, y: cute.Tensor, dst: cute.Tensor):
    threads = 256
    gA = cute.zipped_divide(x, (1, 4))
    gB = cute.zipped_divide(y, (1, 4))
    gC = cute.zipped_divide(dst, (1, 4))
    cute.printf(gC.shape, gC.layout)
    vectorized_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads, 1, 1),
        block=(threads, 1, 1),
    )

M, N = 256, 256
a = torch.randn(M, N, dtype=torch.float32, device="cuda")
b = torch.randn(M, N, dtype=torch.float32, device="cuda")
c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

vadd_compiled = cute.compile(vectorized_add, from_dlpack(a), from_dlpack(b), from_dlpack(c))
vadd_compiled(from_dlpack(a), from_dlpack(b), from_dlpack(c))

c_torch = a + b
print(torch.allclose(c_torch, c))
