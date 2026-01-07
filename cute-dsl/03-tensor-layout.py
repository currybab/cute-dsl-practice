import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

print("tensor_demo")

@cute.jit
def tensor_demo(t: cute.Tensor):
    cute.printf("t[0,0] = {}", t[0, 0])
    sub = t[(None, 0)]   # First row view
    frag = cute.make_rmem_tensor(sub.layout, sub.element_type)
    frag.store(sub.load())
    cute.print_tensor(frag)

arr = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
print(arr.shape)
tensor_demo(from_dlpack(arr))
print("====================\n\n")

print("layout_stride_demo")

@cute.jit
def layout_stride_demo(M: cutlass.Int32, N: cutlass.Int32):
    row_major = cute.make_layout((M, N), stride=(N, cutlass.Int32(1)))
    col_major = cute.make_layout((M, N), stride=(cutlass.Int32(1), M))
    print("static row-major:", row_major)
    print("static col-major:", col_major)
    cute.printf("dynamic row-major: {}", row_major)
    cute.printf("dynamic col-major: {}", col_major)

layout_stride_demo(cutlass.Int32(4), cutlass.Int32(3))
print("====================\n\n")

print("slicing_examples")

@cute.jit
def slicing_examples(t: cute.Tensor):
    # scalar access
    cute.printf("t[1,2] = {}", t[1, 2])

    # Entire second row (shape: (N,)) using (None, row_index)
    row = t[(1, None)]
    cute.printf("row = {}", row)
    row_frag = cute.make_rmem_tensor(row.layout, row.element_type)
    row_frag.store(row.load())
    print("Second row:")
    cute.print_tensor(row_frag)

    # Entire third column (shape: (M,)) using (col_index, None)
    col = t[None, 2]
    col_frag = cute.make_rmem_tensor(col.layout, col.element_type)
    col_frag.store(col.load())
    print("Third column:")
    cute.print_tensor(col_frag)

    # Printing the first row directly (*t[2] == *t[2, 0])
    cute.printf(
        "t[2] = {} (equivalent to t[{}])",
        t[2],
        cute.make_identity_tensor(t.layout.shape)[2]
    )
    cute.printf("layout: {}, shape: {}", t.layout, t.layout.shape)


# 4x3 example tensor
arr = torch.arange(12, dtype=torch.float32).reshape(4, 3)
slicing_examples(from_dlpack(arr))
print("====================\n\n")
