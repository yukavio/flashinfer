import torch
from flashinfer.gemm import get_gemm_module

module = get_gemm_module()

m = 128
n = 128
k = 128

a = torch.randn(m, k, device='cuda', dtype=torch.half)
b = torch.randn(k, n, device='cuda', dtype=torch.half)
c = [torch.randn(m, n, device='cuda', dtype=torch.half)]
print(c)
module.cutlass_gemm_broadcast(a, b, c, 0)

