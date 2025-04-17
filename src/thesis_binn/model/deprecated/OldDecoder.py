import torch
import torch.nn as nn
from thesis_binn.data_processing.ProxyTerm import ProxyTerm
from thesis_binn.model.Coder import DenseCoder
from thesis_binn.model.SparseLinear import SparseLinear


class Decoder2(DenseCoder):
    def __init__(self, go_layers, dtype):
        self.activation = nn.ReLU()

        super(DenseCoder, self).__init__(go_layers, self.activation, dtype)
        # Option to add a final activation before ModuleList conversion
        self.net_layers = nn.ModuleList(self.net_layers)

        # (Re)initialize all weights
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


class Decoder(nn.Module):
    """Base class for fully connected dense decoder."""

    def __init__(self, go_layers, dtype):
        super(Decoder, self).__init__()

        # Initialize architecture
        self.go_layers = go_layers
        network_layers = []
        self.activation = nn.ReLU()
        for i in range(len(self.go_layers) - 1):
            network_layers.append(nn.Linear(len(self.go_layers[i]), len(self.go_layers[i + 1]), dtype=dtype))
            if i < len(self.go_layers) - 2:
                network_layers.append(self.activation)
            else:
                # Option to add a final activation layer (used to be Sigmoid, but was removed for better performance)
                pass
        self.layers = nn.ModuleList(network_layers)

        # Initialize all weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def mask_weights(self):
        """Implemented in BIDecoder subclass."""
        return

    def mask_gradients(self):
        """Implemented in BIDecoder subclass."""
        return


class BIDecoder(Decoder):
    """Decoder subclass. Uses dense weights and dense masks derived from provided GO structure to construct a biologically-informed dense decoder."""

    def __init__(self, go_layers, dtype, masks=None):
        super(BIDecoder, self).__init__(go_layers, dtype)

        # Initialize biologically-informed masks
        if masks:
            self.edge_masks = masks[0]
            self.proxy_masks = masks[1]
        else:
            self.proxy_masks = self._create_proxy_masks()
            self.edge_masks = self._create_edge_masks()

        # Mask weights using proxy and edge masks
        self.mask_weights()

    def mask_weights(self):
        """Using the internal dense mask matrices, mask the dense weights and biases after each training step."""
        # Mask weights using proxy and edge masks
        mask_index = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Set weights towards proxies to 1, bias towards proxies to 0
                layer.weight.data = torch.masked_fill(layer.weight.data, self.proxy_masks[mask_index], value=1)
                layer.bias.data = torch.masked_fill(layer.bias.data, self.proxy_masks[mask_index].T, value=0)
                # Set weights without edges to 0
                layer.weight.data = torch.masked_fill(layer.weight.data, self.edge_masks[mask_index] == 0, value=0)
                mask_index += 1

    def mask_gradients(self):
        """Mask the dense gradients of weights and biases after the backwards pass to prevent masked values getting updates."""
        # Mask gradients using proxy and edge masks
        mask_index = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Set gradients of proxy weights and biases to 0
                layer.weight.grad = torch.masked_fill(layer.weight.grad, self.proxy_masks[mask_index], value=0)
                layer.bias.grad = torch.masked_fill(layer.bias.grad, self.proxy_masks[mask_index].T, value=0)
                # Set gradients of weights without edges to 0
                layer.weight.grad = torch.masked_fill(layer.weight.grad, self.edge_masks[mask_index] == 0, value=0)
                mask_index += 1

    def _create_proxy_masks(self):
        """Returns a list of dense 1D boolean tensors that represent each network layer. Each non-zero entry means that the corresponding term in that layer is a ProxyTerm."""
        proxy_masks = []
        for n in range(len(self.go_layers) - 1):
            next_layer = self.go_layers[n + 1]
            proxy_mask = torch.zeros(len(next_layer), 1, dtype=torch.bool)
            for i in range(len(next_layer)):
                if isinstance(next_layer[i], ProxyTerm):
                    proxy_mask[i] = 1
            proxy_masks.append(proxy_mask)
        return proxy_masks

    def _create_edge_masks(self):
        """Returns a list of dense 2D boolean tensors. Each tensor functions as an adjacency matrix between network layers, derived from GO."""
        edge_masks = []
        for n in range(len(self.go_layers) - 1):
            current_layer = self.go_layers[n]
            next_layer = self.go_layers[n + 1]
            edge_mask = torch.zeros(len(next_layer), len(current_layer), dtype=torch.bool)

            for i in range(len(current_layer)):
                parent = current_layer[i]
                child_ids = [child.item_id for child in parent.children]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in child_ids:
                        edge_mask[j][i] = 1
            edge_masks.append(edge_mask)
        return edge_masks


class SparseDecoder(nn.Module):
    def __init__(self, go_layers, dtype, protocol="coo"):
        super(SparseDecoder, self).__init__()
        self.go_layers = go_layers
        self.protocol = protocol

        # Initialize masks
        self.proxy_masks = self._create_proxy_masks()
        self.edge_masks = self._create_edge_masks()

        # Initialize architecture
        network_layers = []
        self.activation = nn.ReLU()
        for i in range(len(self.go_layers) - 1):
            network_layers.append(
                SparseLinear(len(self.go_layers[i]), len(self.go_layers[i + 1]), self.edge_masks[i], dtype=dtype,
                             protocol=self.protocol))
            if i < len(self.go_layers) - 2:
                network_layers.append(self.activation)
            # Final layer has no activation
        self.layers = nn.ModuleList(network_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _create_proxy_masks(self):
        proxy_masks = []
        for n in range(len(self.go_layers) - 1):
            next_layer = self.go_layers[n + 1]
            proxy_mask = torch.zeros(len(next_layer), 1, dtype=torch.bool)
            for i in range(len(next_layer)):
                if isinstance(next_layer[i], ProxyTerm):
                    proxy_mask[i] = 1
            proxy_masks.append(proxy_mask)
        return proxy_masks

    def _create_edge_masks(self):
        edge_masks = []
        for n in range(len(self.go_layers) - 1):
            current_layer = self.go_layers[n]
            next_layer = self.go_layers[n + 1]
            edge_mask = torch.zeros(len(next_layer), len(current_layer), dtype=torch.bool)

            for i in range(len(current_layer)):
                child = current_layer[i]
                parent_ids = [parent.item_id for parent in child.parents]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in parent_ids:
                        edge_mask[j][i] = 1
            edge_masks.append(edge_mask)
        return edge_masks
