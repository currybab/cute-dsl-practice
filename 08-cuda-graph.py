import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream

@cute.kernel
def hello_kernel():
    cute.printf("Hello")

@cute.jit
def hello_launch(stream: CUstream):
    hello_kernel().launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

s = current_stream()
compiled = cute.compile(hello_launch, CUstream(s.cuda_stream))

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    compiled(CUstream(current_stream().cuda_stream))

g.replay()
torch.cuda.synchronize()
g.replay()
