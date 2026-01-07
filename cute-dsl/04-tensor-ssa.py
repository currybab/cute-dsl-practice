import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

print("ssa_add")

@cute.jit
def ssa_add(dst: cute.Tensor, x: cute.Tensor, y: cute.Tensor):
    xv = x.load()
    yv = y.load()
    dst.store(xv + yv)
    cute.print_tensor(dst)

X = np.ones((2, 3), dtype=np.float32)
Y = np.full((2, 3), 2.0, dtype=np.float32)
Z = np.zeros((2, 3), dtype=np.float32)
ssa_add(from_dlpack(Z), from_dlpack(X), from_dlpack(Y))

print("====================\n\n")

print("ssa_reduce")
# op: A cute.ReductionOp enum specifying the operation (ADD, MUL, MAX, MIN, …)
# init: Initial accumulator value (also sets accumulator dtype)
# reduction_profile: Axes to reduce — 0 for all axes; or a tuple with 1 to reduce / None to keep

@cute.jit
def ssa_reduce(a: cute.Tensor):
    v = a.load()
    
    # 1. Total Sum (Scalar는 바로 출력 가능)
    total = v.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile = 0)
    cute.printf("total sum = {}", total)

    # 2. Row-wise sum
    row_sums_vec = v.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile = (None, 1))
    cute.printf("row sums")
    cute.print_tensor(row_sums_vec)
    # cute.printf("row sums = {}", row_sums_vec) # 이상한 똥값 나옴

    # 3. Col-wise sum
    col_sums_vec = v.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=(1, None))
    cute.printf("col sums")
    cute.print_tensor(col_sums_vec)
    # cute.printf("col sums = {}", col_sums_vec)

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
ssa_reduce(from_dlpack(A))
