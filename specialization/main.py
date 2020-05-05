import torch
import time
torch.manual_seed(0)
from benchmark.test_cases import test_cases

equation = "ij,jk->ik"
lhs, rhs = equation.split("->")
ops = lhs.split(",")
ops.append(rhs)

# for e in test_cases:
#     print(e.equation, e.native_func)
op_mapping = {
    repr(['a','']): torch.sum,
    repr(['a', 'a', '']): torch.dot,
    repr(['a', 'a', 'a']): torch.mul,
    repr(['a', 'b', 'ab']): torch.ger,
    repr(['ab', 'ba']): lambda e: torch.transpose(e, 0, 1),
    repr(['ab', 'b']): lambda e: torch.sum(e, dim=0),
    repr(['ab', 'a']): lambda e: torch.sum(e, dim=1),
    repr(['ab', 'ab','ab']): torch.mul,
    repr(['ab', 'b', 'a']): torch.mv,
    repr(['ab', 'bc', 'ac']): torch.matmul,
    repr(['ab', 'bc', 'cd', 'ad']): torch.chain_matmul,
}
def parse_input(input_list):
    """ Parse the input into 
        ij,jk->ik => ab,bc->ac
        ab,bc,cd
        ...
    """
    parse_dict = {}
    output_str = [''] * len(input_list)
    for index in range(len(input_list)):
        input_dim = input_list[index]
        for dim in input_dim:
            if dim not in parse_dict.keys():
                parse_dict[dim] = len(parse_dict.keys())
                
            output_str[index] = output_str[index] + chr(ord('a') + parse_dict[dim])
    return output_str



def einsum_spec(input_string, tensor_list):
    input_dims, output = input_string.split('->')
    input_dim = len(input_dims.split(',')[0])
    
    lhs = input_dims.split(',')
    lhs.append(output)
    parsed = parse_input(lhs)
    return op_mapping[repr(parsed)](*tensor_list)

if __name__ == "__main__":
    a = torch.rand(50000)
    # einsum_spec('i->', [a])
    start = time.time()
    einsum_spec('i,i->', [a, a])
    end = time.time()
    print(end - start)
    start = time.time()
    torch.einsum('i,i->', a,a)
    end = time.time()
    print(end - start)
    # einsum_spec('i,i->i', [a, a])
    # einsum_spec('i,j->ij', [a, a])
    
    b = torch.rand(500, 500) 
    einsum_spec('ij->ji', [b])
    einsum_spec('ij->j', [b])
    einsum_spec('ij->i', [b])
    einsum_spec('ij,ij->ij', [b, b])
    einsum_spec('ij,j->i', [b, a])
    einsum_spec('ij,jk->ik', [b, b])

    einsum_spec('ab,bc,cd->ad', [b, b, b,b])
    # einsum_spec('ij, jk->', [b, b])