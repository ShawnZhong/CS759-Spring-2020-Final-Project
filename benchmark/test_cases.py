import torch

from tester import Tester

# https://github.com/pytorch/pytorch/pull/6307/files#diff-9996665f82f52030836eb8657057cfadR1312
test_cases = [
    # Vector
    Tester(torch.sum, "i->", "sum"),
    Tester(torch.dot, "i,i->", "dot"),
    Tester(torch.mul, "i,i->i", "vector element-wise mul"),
    Tester(torch.ger, "i,j->ij", "outer"),
    # Matrix
    Tester(lambda e: torch.transpose(e, 0, 1), "ij->ji", "transpose"),
    Tester(lambda e: torch.sum(e, dim=0), "ij->j", "row sum"),
    Tester(lambda e: torch.sum(e, dim=1), "ij->i", "col sum"),
    Tester(torch.mul, "ij,ij->ij", "matrix element-wise mul"),
    Tester(torch.mv, "ij,j->i", "matrix vector multiplication"),
    Tester(torch.matmul, "ij,jk->ik", "matmul"),
    # Tensor
    Tester(torch.bmm, "aij,ajk->aik", "batch matmul"),
    Tester(torch.chain_matmul, "ab,bc,cd->ad", "chain matmul"),
]
