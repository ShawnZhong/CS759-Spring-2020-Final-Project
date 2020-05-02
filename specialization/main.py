import torch

from benchmark.test_cases import test_cases
def parse_input(input_list):
    """ Parse the input into 
        ab,bc
        ab,bc,cd
        ...
    """
    parse_dict = {}
    for input_dim in input_list:
        for 

def einsum_spec(input_string, tensor_list):
    input_dims, output = input_string.split('->')
    input_dim = len(input_dims.split(',')[0])
    parsed = parse_input(input_dims.split(','))
    if len(output) == 0: 
        # just summing the elements
        if len(input_dims.split(',')) == 1:
            return torch.sum(tensor_list[0])
        assert len(input_dims.split(',')) == len(tensor_list)
        assert len(input_dims.split(',')[0]) == len(tensor_list[0].shape) 
        if len(tensor_list[0].shape) == 1:
            return torch.dot(tensor_list[0])
        

        
    output_dim = len(output.split(',')[0])
    


if __name__ == "__main__":
    a = torch.rand(500)
    einsum_spec('i->', [a])
    b = torch.rand(500, 500) 
    einsum_spec('ij, jk->', [b, b])

