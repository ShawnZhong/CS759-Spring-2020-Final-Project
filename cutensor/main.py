import cupy
from cupy import cutensor
import time
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from cupy.cuda import stream

# 'abcd,aefd->aefbc'

batch_dim = 200

def einsum_cutensor(n):
    st = stream.Stream()
    a = cupy.random.rand(batch_dim, n, n, n)
    b = cupy.random.rand(batch_dim, n, n, n)

    arr_out = cupy.empty([batch_dim, n, n, n, n])
    arr0 = cupy.ascontiguousarray(a)
    arr1 = cupy.ascontiguousarray(b)
    desc_0 = cutensor.create_tensor_descriptor(arr0)
    desc_1 = cutensor.create_tensor_descriptor(arr1)
    desc_out = cutensor.create_tensor_descriptor(arr_out)
    with st:
        arr_out = cutensor.contraction(1.0,
                                    arr0, desc_0, list(ord(c) for c in "abcd"),
                                    arr1, desc_1, list(ord(c) for c in "aefd"),
                                    0.0,
                                    arr_out, desc_out, list(ord(c) for c in "aefbc"))
    st.synchronize()


def einsum_pytorch(n):
    a = torch.rand(batch_dim, n, n, n)
    b = torch.rand(batch_dim, n, n, n)
    torch.einsum("abcd,aefd->aefbc", a, b)
    torch.cuda.synchronize()

def benchmark(func, n):
    for _ in range(3):
        func(n)
    start = time.time()
    for _ in range(10):
        func(n)
    end = time.time()
    return (end - start) / 10


def main():
    output_dir = Path(__file__).parent
    os.makedirs(output_dir, exist_ok=True)

    xs = range(2, 31, 2)

    for func in (einsum_cutensor, einsum_pytorch):
        times = [benchmark(func, n) for n in xs]
        plt.plot(xs, times, label=func.__name__)
    

    # title = f"{test_case}"
    # plt.title(title)
    # plt.yscale('log', basey=10)
    # plt.xscale('log', basex=2)
    plt.xticks(xs)
    plt.xlabel("n")
    plt.ylabel("Time (ms)")
    plt.legend()

    plt.savefig(output_dir / "result.png", dpi=300)
    plt.show(block=False)
    plt.close()

if __name__ == "__main__":
    main()