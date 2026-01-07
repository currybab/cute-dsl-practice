import cutlass
import cutlass.cute as cute

@cute.jit
def layout_demo():
    # Composition and coalesce
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2))
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)
    C = cute.coalesce(R)

    # Logical divide with tiler
    L = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))
    T = (cute.make_layout(3, stride=3),
         cute.make_layout((2, 4), stride=(1, 8)))
    D = cute.logical_divide(L, tiler=T)

    # Logical product/repetition
    P = cute.logical_product(
        cute.make_layout((2, 2), stride=(4, 1)),
        cute.make_layout(6, stride=1),
    )

    cute.printf("A={}, B={}, R={}, C={}", A, B, R, C)
    cute.printf("Divide: {}", D)
    cute.printf("Product: {}", P)

layout_demo()
