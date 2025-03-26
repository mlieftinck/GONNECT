import torch
import torch.nn as nn

from model.Autoencoder import Autoencoder
from model.Decoder import Decoder, DenseBIDecoder
from model.Encoder import DenseBIEncoder


class MSE(nn.Module):
    # Wrapper class to allow additional arguments through **kwargs, required for different loss functions
    def __init__(self, model=None):
        super(MSE, self).__init__()
        self.name = "MSE Loss"
        self.mse = nn.MSELoss()

    def forward(self, x, y, **kwargs):
        return self.mse(x, y)


class MSE_Soft_Link_Sum(nn.Module):
    def __init__(self, model=None):
        super(MSE_Soft_Link_Sum, self).__init__()
        self.name = "MSE + soft link sum"
        self.model = model

    def forward(self, x, y, model: Autoencoder, alpha=1.0, **kwargs):
        mse = nn.functional.mse_loss(x, y)
        soft_weight_sum = 0
        if hasattr(model.encoder, "edge_masks"):
            soft_weight_sum += soft_link_sum(model.encoder)
        if hasattr(model.decoder, "edge_masks"):
            soft_weight_sum += soft_link_sum(model.decoder)
        # print("soft_weight_sum: ", soft_weight_sum)
        return mse + alpha * soft_weight_sum


def soft_link_sum(net):
    """For a given network (encoder or decoder), return the sum of absolute values of the weights that are considered soft links because they are masked by the edge mask of the network."""
    soft_weight_sum = 0
    mask_index = 0
    for layer in net.net_layers:
        if isinstance(layer, nn.Linear):
            # For each linear layer of the network, sum the absolute values of the masked weights
            soft_weight_sum += torch.sum(layer.weight.abs() * ~net.edge_masks[mask_index])
            mask_index += 1
    return soft_weight_sum


if __name__ == '__main__':
    loss_fn = MSE_Soft_Link_Sum()
    layers = torch.randn((2, 3))
    mask_e = [torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]).bool()]
    encoder_weights = torch.Tensor([
        [1, 2, 3],
        [4, -5, 6],
        [7, 8, 9]])
    masks = [mask_e, [torch.randn(3, 1) > 0.5]]
    encoder = DenseBIEncoder(layers, nn.ReLU(), torch.float64, masks)
    encoder.net_layers[0].weight.data = encoder_weights
    print(f"Test result should be 15. Result: {soft_link_sum(encoder)}")

    decoder = Decoder(layers, nn.ReLU(), torch.float64)
    ae = Autoencoder(encoder, decoder)
    x = torch.Tensor([1, 2, 3]).requires_grad_()
    print(
        f"Test result should be 15. Result: {loss_fn(x, torch.Tensor([1, 2, 3]), ae)}")
    print(
        f"Test result should be 16. Result: {loss_fn(x, torch.Tensor([0, 1, 2]), ae)}")
    decoder = DenseBIDecoder(layers, nn.ReLU(), torch.float64, masks)
    decoder.net_layers[0].weight.data = encoder_weights
    ae = Autoencoder(encoder, decoder)
    print(
        f"Test result should be 30. Result: {loss_fn(x, torch.Tensor([1, 2, 3]), ae)}")
    print(
        f"Test result should be 31. Result: {loss_fn(x, torch.Tensor([0, 1, 2]), ae)}")
