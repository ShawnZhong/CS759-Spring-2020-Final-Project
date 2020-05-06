import cupy
from cupy import cutensor
import time
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 'abcd,aefd->aefbc'

def einsum_cutensor(n):
    a = cupy.random.rand(n, n, n, n)
    b = cupy.random.rand(n, n, n, n)

    arr_out = cupy.empty([n, n, n, n, n], cupy.float64)
    arr0 = cupy.ascontiguousarray(a)
    arr1 = cupy.ascontiguousarray(b)
    desc_0 = cutensor.create_tensor_descriptor(arr0)
    desc_1 = cutensor.create_tensor_descriptor(arr1)
    desc_out = cutensor.create_tensor_descriptor(arr_out)
    arr_out = cutensor.contraction(1.0,
                                arr0, desc_0, list(ord(c) for c in "abcd"),
                                arr1, desc_1, list(ord(c) for c in "aefd"),
                                0.0,
                                arr_out, desc_out, list(ord(c) for c in "aefbc"))


def einsum_pytorch(n):
    a = torch.rand(n, n, n, n)
    b = torch.rand(n, n, n, n)
    torch.einsum("abcd,aefd->aefbc", a, b)

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

    xs = range(2, 49, 2)

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