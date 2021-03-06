import itertools
import torch
import time
import matplotlib.pyplot as plt


# Get minimal representation of a string
def getminimal(terms):
    c = 'a'
    skip = [',', "-", ">"]
    dict = {x : x for x in skip}
    for x in terms:
        if x not in skip and x not in dict:
            dict[x] = c
            c = chr(ord(c) + 1)
    return ''.join(dict[c] for c in terms)

def get_mask(term):
    return sum(1 << (ord(c) - ord('a')) for c in term)

def evaluate_path(terms):
    terms = getminimal(terms)
    [prefix, suffix] = terms.split("->")
    vals = [get_mask(term) for term in [*prefix.split(","), suffix]]
    n = len(vals)
    pmask = vals.copy()
    for i in range(1, n):
        pmask[i] = pmask[i-1] | pmask[i]
    smask = vals.copy()
    for i in reversed(range(0, n - 1)):
        smask[i] = smask[i+1] | smask[i]
    res = 0
    for i in range(n - 1):
        p = 0 if i == 0 else pmask[i - 1]
        s = smask[i + 1]
        curmask = (p & s) | vals[i]
        res = max(res, bin(curmask).count("1"))
    return res

def find_opt_path_brute_force(terms):
    [prefix, suffix] = terms.split("->")
    ops = prefix.split(",")
    idx = range(len(ops))
    min_score = -1
    permute = -1
    for per in list(itertools.permutations(idx)):
        path = "->".join([','.join(ops[per[i]] for i in range(len(ops))), suffix])
        # print(path)
        cur_score = evaluate_path(path)

        if min_score == -1 or cur_score < min_score:
            min_score = cur_score
            permute = per
    return (min_score, permute)

def find_opt_path_dp(terms):
    terms = getminimal(terms)
    [prefix, suffix] = terms.split("->")
    vals = [get_mask(term) for term in prefix.split(",")]
    final_mask = get_mask(suffix)
    n = len(vals)
    sum_mask = [0] * (1 << n)
    for mask in range(1 << n):
        for j in range(n):
            if ((mask >> j) & 1) == 1:
                sum_mask[mask] |= vals[j]
    dp = [-1] * (1 << n)
    back = [-1] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        for j in range(n):
            if ((mask >> j) & 1) == 0:
                smask = (sum_mask[((1 << n) - 1) ^ (mask | (1 << j))] | final_mask)
                nxtdp = max(dp[mask], bin((sum_mask[mask] & smask) | vals[j]).count("1"))
                if dp[mask | (1 << j)] == -1 or nxtdp < dp[mask | (1 << j)]:
                    dp[mask | (1 << j)] = nxtdp
                    back[mask | (1 << j)] = j
    permute = []
    cur = (1 << n) - 1
    while cur != 0:
        permute.append(back[cur])
        cur ^= (1 << back[cur])
    return (dp[(1 << n) - 1], tuple(reversed(permute)))

def permute_terms(terms, per):
    [prefix, suffix] = terms.split("->")
    ops = prefix.split(",")
    return "->".join([','.join(ops[per[i]] for i in range(len(ops))), suffix])

def fast_einsum(equation, operands):
    [_, p] = find_opt_path_dp(equation)
    new_equation = permute_terms(equation, p)
    print("fast_einsum using new equation: " + new_equation)
    return torch.einsum(new_equation, [operands[i] for i in p])

def benchmark(equation, operands):
    print("benchmarking on equation " + equation + "\n")
    total_einsum = 0
    total_fast = 0
    for i in range(10):
        print("torch.einsum: ")
        einsum_time = -time.perf_counter()
        torch.einsum(equation, operands)
        einsum_time += time.perf_counter()
        if i >= 5:
            total_einsum += einsum_time

        print("fast_einsum:")
        fast_time = -time.perf_counter()
        fast_einsum(equation, operands)
        fast_time += time.perf_counter()
        if i >= 5:
            total_fast += fast_time
    return total_einsum / 5, total_fast / 5


def plot_time(xs, fast_ein_time, ein_time, plot_title):
    plt.plot(xs, fast_ein_time, label='fast einsum')
    plt.plot(xs, ein_time, label='Pytorch einsum')
    title = plot_title
    plt.title(title)
    plt.yscale('log', basey=10)
    plt.xticks(xs)
    plt.xlabel("n")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.savefig( f"./path_opt/results/{title}.png")
    plt.show(block=False)
    plt.close() 

y1s = []
y1_ein = []
y2s = []
y2_ein = []
for dim in range(1, 20):
    einsum_time, fast_time = benchmark('bdik,acaj,ikab,ajac,ikbd->abjk', [torch.rand(dim, dim, dim, dim) for i in range(5)])
    y1_ein.append(einsum_time)
    y1s.append(fast_time)
    einsum_time, fast_time = benchmark("ab,cd,ef,fe,dc,ba->a", [torch.rand(dim, dim), torch.rand(dim, dim),
                                    torch.rand(dim, dim), torch.rand(dim, dim),
                                    torch.rand(dim, dim), torch.rand(dim, dim)])
    y2_ein.append(einsum_time)
    y2s.append(fast_time)


x = [i for i in range(1, 20)]
plot_time(x, y1s, y1_ein, "bdik,acaj,ikab,ajac,ikbd->abjk")
plot_time(x, y2s, y2_ein, "ab,cd,ef,fe,dc,ba->a")