import torch
import time
torch.manual_seed(0)
import matplotlib.pyplot as plt

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
    repr(['abc', 'acd', 'abd']): torch.bmm,
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

def tester_plot(input_string,  use_cuda, plot_title):
    device = torch.device("cuda" if use_cuda else "cpu")
    ys = []
    ys_einsum= []
    if use_cuda:
        for i in range(10):
            
            dim_list = [len(e) for e in input_string.split('->')[0].split(",")]
            size_list = []
            
            for dim in dim_list:
                size_list.append([2**i for _ in range(dim)])
            
            tensor_list = [torch.rand(size, device=device) for size in size_list]
            
            total_time = 0
            total_time_einsum = 0
            for j in range(10):
                start = time.time()
                result = einsum_spec(input_string, tensor_list)
                torch.cuda.synchronize(device)
                end = time.time()
                if j >= 5: 
                    total_time += (end - start)

                start = time.time()
                result = torch.einsum(input_string, *tensor_list)
                torch.cuda.synchronize(device)
                end = time.time()
                if j >= 5:
                    total_time_einsum += (end - start)
            ys.append(total_time / 5)
            ys_einsum.append(total_time_einsum/5)
        xs = [pow(2, i) for i in range(10)]
        plt.plot(xs, ys, label='native')
        plt.plot(xs, ys_einsum, label='einsum')
        title = plot_title
        plt.title(title)
        plt.yscale('log', basey=10)
        plt.xscale('log', basex=2)
        plt.xticks(xs)
        plt.xlabel("n")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.savefig( f"./specialization/results/{title}.png")
        plt.show(block=False)
        plt.close() 
    

if __name__ == "__main__":



    # compare GPU 
    device = torch.device("cuda")
    tester_plot('i->', True, "sum")
    tester_plot('i,i->', True, "dot")
    tester_plot("i,i->i", True, "vector element-wise mul")
    tester_plot("i,j->ij", True, "outer")
    # Matrix
    tester_plot("ij->ji", True, "transpose")
    tester_plot( "ij->j", True, "row sum")
    tester_plot("ij->i", True, "col sum")
    tester_plot("ij,ij->ij", True, "matrix element-wise mul")
    tester_plot( "ij,j->i", True, "matrix vector multiplication")
    tester_plot("ij,jk->ik", True, "matmul")
    # Tensor
    tester_plot("aij,ajk->aik", True, "batch matmul")
    tester_plot("ab,bc,cd->ad", True, "chain matmul")

    
    
        


    




