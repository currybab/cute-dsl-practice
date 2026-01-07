import cutlass
import cutlass.cute as cute

@cute.jit
def print_demo(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print("static a:", a)   # => ? (dynamic)
    print("static b:", b)   # => 2
    cute.printf("dynamic a: {}", a)
    cute.printf("dynamic b: {}", b)
    layout = cute.make_layout((a, b))
    print("static layout:", layout)       # (?,2):(1,?)
    cute.printf("dynamic layout: {}", layout)  # (8,2):(1,8)

print_demo(cutlass.Int32(8), 2)

@cute.jit
def dtypes():
    a = cutlass.Int32(42)
    b = a.to(cutlass.Float32)
    c = b + 0.5
    d = c.to(cutlass.Int32)
    cute.printf("a={}, b={}, c={}, d={}", a, b, c, d)

dtypes()
