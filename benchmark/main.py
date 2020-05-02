import os
from itertools import takewhile, count
from pathlib import Path

import matplotlib.pyplot as plt

from .test_cases import test_cases


def plot(xs, profs):
    for method, prof in zip(("einsum", "native"), zip(*profs)):
        times = [e.self_cpu_time_total for e in prof]
        for x, y in zip(xs, times):
            plt.text(
                x, y, f"{y:.0f}",
                horizontalalignment='center',
            )
        plt.plot(xs, times, label=method)


def main():
    output_dir = Path(__file__).parent / "result"
    os.makedirs(output_dir, exist_ok=True)

    for test_case in test_cases:
        for use_cuda in (True,):
            sizes_gen = (2 ** i for i in count(1))
            profs_gen = ((size, test_case.profile(size, use_cuda)) for size in sizes_gen)
            sizes, profs = zip(*takewhile(lambda x: x[1], profs_gen))

            plot(sizes, profs)

            title = f"{test_case}"
            plt.title(title)
            plt.yscale('log', basey=10)
            plt.xscale('log', basex=2)
            plt.xticks(sizes)
            plt.xlabel("n")
            plt.ylabel("Time (ms)")
            plt.legend()

            plt.savefig(output_dir / f"{title}.png")
            plt.show(block=False)
            plt.close()


if __name__ == "__main__":
    main()