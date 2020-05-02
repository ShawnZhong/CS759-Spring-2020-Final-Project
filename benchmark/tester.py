from typing import Optional, Tuple

import torch


class Tester:
    def __init__(self, native_func, equation, name) -> None:
        super().__init__()
        self.native_func = native_func
        self.equation = equation
        self.name = name

        input_str, _ = self.equation.split("->")
        self.dims = [len(e) for e in input_str.split(",")]

    @staticmethod
    def run_profiler(func, args) -> Optional[Tuple[torch.autograd.profiler.profile, torch.Tensor]]:
        for _ in range(3):
            func(*args)

        result = None
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for _ in range(10):
                result = func(*args)
        return prof, result

    def profile(self, n: int, use_cuda: bool = False):
        device = torch.device("cuda" if use_cuda else "cpu")
        sizes = [(n,) * dim for dim in self.dims]

        try:
            tensors = [torch.rand(size, device=device) for size in sizes]
            prof_einsum, result_einsum = self.run_profiler(torch.einsum, (self.equation, *tensors))
            prof_native, result_native = self.run_profiler(self.native_func, tensors)
            assert torch.allclose(result_native, result_einsum)
        except RuntimeError:
            return None

        print(
            f"{self.name:<30}{device.type:<8}{str(sizes):<32}"
            f"{prof_einsum.self_cpu_time_total:12.3f}"
            f"{prof_native.self_cpu_time_total:12.3f}",
        )

        return prof_einsum, prof_native

    def __str__(self) -> str:
        def tensor_to_str(dim):
            dim_str = ','.join('n' * dim)
            return f"Tensor({dim_str})"

        return f"{self.name} with {', '.join(tensor_to_str(dim) for dim in self.dims)}"
