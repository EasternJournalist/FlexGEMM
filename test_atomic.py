import torch
import triton
import triton.language as tl
from flex_gemm.kernels.triton.hashmap import _masked_atomic_cas_b32

@triton.jit
def test_kernel(out_ptr, prev_ptr, mask_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    m = tl.load(mask_ptr + idx).to(tl.int1)
    # All lanes target slot 0, all desire idx+100, all expect -1
    prev = _masked_atomic_cas_b32(out_ptr + 0, -1, idx + 100, mask=m)
    tl.store(prev_ptr + idx, prev)

def run():
    N = 8
    out = torch.full((1,), -1, dtype=torch.int32, device='cuda')
    prev = torch.zeros((N,), dtype=torch.int32, device='cuda')
    # Only lanes 2 and 5 active
    m = torch.tensor([0,0,1,0,0,1,0,0], dtype=torch.int8, device='cuda')
    test_kernel[(1,)](out, prev, m, N=N)
    print("out =", out.tolist())
    print("prev =", prev.tolist())
    print("expected: out has either 102 or 105; prev shows -1 for inactive lanes, -1 for winner, winner-value for loser")

    # Repeat: all active
    out2 = torch.full((1,), -1, dtype=torch.int32, device='cuda')
    prev2 = torch.zeros((N,), dtype=torch.int32, device='cuda')
    m2 = torch.ones((N,), dtype=torch.int8, device='cuda')
    test_kernel[(1,)](out2, prev2, m2, N=N)
    print("\nall-active:")
    print("out =", out2.tolist())
    print("prev =", prev2.tolist())

if __name__ == "__main__":
    run()
