import math
import torch
import torch.nn as nn


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, mask: torch.Tensor, bias=True, device=None, dtype=None,
                 protocol="coo"):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edge_mask = mask
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if protocol == "coo":
            self.protocol = torch.sparse_coo
            self.matrix_multiplication = self.mm_coo
        elif protocol == "csr":
            self.protocol = torch.sparse_csr
            self.matrix_multiplication = self.mm_csr
        else:
            raise Exception(f"Unknown sparsity protocol: {protocol}")

        # Construct sparse weights according to edge mask
        nnz_indices = self.edge_mask.indices()
        nnz_values = torch.zeros_like(self.edge_mask.values(), dtype=dtype)
        self.weight = nn.Parameter(torch.sparse_coo_tensor(nnz_indices, nnz_values))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Adapted from nn.Linear to be able to handle sparsity."""
        self.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix_multiplication(x, self.weight.T) + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    @staticmethod
    def mm_coo(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(x, w)

    @staticmethod
    def mm_csr(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.matmul(x)

    ####################################################################################################################
    #   The methods below are copied from the nn.Linear module. They are adapted to be able to handle sparse Tensors.  #
    ####################################################################################################################
    def kaiming_uniform_(
            self,
            tensor,
            a: float = 0,
            mode: str = "fan_in",
            nonlinearity: str = "leaky_relu",
            generator=None,
    ):
        r"""Fill the input `Tensor` with values using a Kaiming uniform distribution.

        The method is described in `Delving deep into rectifiers: Surpassing
        human-level performance on ImageNet classification` - He, K. et al. (2015).
        The resulting tensor will have values sampled from
        :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

        .. math::
            \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

        Also known as He initialization.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            a: the negative slope of the rectifier used after this layer (only
                used with ``'leaky_relu'``)
            mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
                backwards pass.
            nonlinearity: the non-linear function (`nn.functional` name),
                recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
            generator: the torch Generator to sample from (default: None)

        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

        Note:
            Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
            that the weight matrix is used in a transposed manner,
            (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
            This is important for correct initialization.
            If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
            pass in a transposed weight matrix, i.e. ``nn.init.kaiming_uniform_(w.T, ...)``.
        """
        # if torch.overrides.has_torch_function_variadic(tensor):
        #     return torch.overrides.handle_torch_function(
        #         kaiming_uniform_,
        #         (tensor,),
        #         tensor=tensor,
        #         a=a,
        #         mode=mode,
        #         nonlinearity=nonlinearity,
        #         generator=generator,
        #     )

        # if 0 in tensor.shape:
        #     warnings.warn("Initializing zero-element tensors is a no-op")
        #     return tensor
        fan = self._calculate_correct_fan(tensor, mode)
        gain = self.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            tensor.data = tensor.data.coalesce()
            return tensor.data.values().uniform_(-bound, bound, generator=generator)

    def _calculate_correct_fan(self, tensor, mode):
        mode = mode.lower()
        valid_modes = ["fan_in", "fan_out"]
        if mode not in valid_modes:
            raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == "fan_in" else fan_out

    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )

        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            # math.prod is not always available, accumulate the product manually
            # we could use functools.reduce but that is not supported by TorchScript
            for s in tensor.shape[2:]:
                receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    @staticmethod
    def calculate_gain(nonlinearity, param=None):
        r"""Return the recommended gain value for the given nonlinearity function.

        The values are as follows:

        ================= ====================================================
        nonlinearity      gain
        ================= ====================================================
        Linear / Identity :math:`1`
        Conv{1,2,3}D      :math:`1`
        Sigmoid           :math:`1`
        Tanh              :math:`\frac{5}{3}`
        ReLU              :math:`\sqrt{2}`
        Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
        SELU              :math:`\frac{3}{4}`
        ================= ====================================================

        .. warning::
            In order to implement `Self-Normalizing Neural Networks`_ ,
            you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
            This gives the initial weights a variance of ``1 / N``,
            which is necessary to induce a stable fixed point in the forward pass.
            In contrast, the default gain for ``SELU`` sacrifices the normalization
            effect for more stable gradient flow in rectangular layers.

        Args:
            nonlinearity: the non-linear function (`nn.functional` name)
            param: optional parameter for the non-linear function

        Examples:
            >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

        .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
        """
        linear_fns = [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        ]
        if nonlinearity in linear_fns or nonlinearity == "sigmoid":
            return 1
        elif nonlinearity == "tanh":
            return 5.0 / 3
        elif nonlinearity == "relu":
            return math.sqrt(2.0)
        elif nonlinearity == "leaky_relu":
            if param is None:
                negative_slope = 0.01
            elif (
                    not isinstance(param, bool)
                    and isinstance(param, int)
                    or isinstance(param, float)
            ):
                # True/False are instances of int, hence check above
                negative_slope = param
            else:
                raise ValueError(f"negative_slope {param} not a valid number")
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        elif nonlinearity == "selu":
            return (
                    3.0 / 4
            )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
        else:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}")
