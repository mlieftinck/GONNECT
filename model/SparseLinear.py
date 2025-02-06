import torch
import torch.nn as nn


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask: torch.Tensor, bias=True, device=None, dtype=None,
                 protocol="coo"):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        self.matrix_multiplication = self.mm_coo
        if protocol == "coo":
            self.protocol = torch.sparse_coo
        elif protocol == "csr":
            self.protocol = torch.sparse_csr
            self.matrix_multiplication = self.mm_csr
        elif protocol == "csc":
            self.protocol = torch.sparse_csc
        elif protocol == "bsr":
            self.protocol = torch.sparse_bsr
        elif protocol == "bsc":
            self.protocol = torch.sparse_bsc
        else:
            raise Exception(f"Unknown sparsity protocol: {protocol}")

        self.edge_mask = mask
        masked_weight = torch.masked_fill(self.weight.data, self.edge_mask == 0, value=0)
        sparse_weight = masked_weight.to_sparse(layout=self.protocol)
        self.weight = nn.Parameter(sparse_weight)
        if bias:
            self.bias.data.to_sparse(layout=self.protocol)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix_multiplication(x, self.weight.T) + self.bias

    @staticmethod
    def mm_coo(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(x, w)

    @staticmethod
    def mm_csr(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.matmul(x)
