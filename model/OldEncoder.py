import torch
import torch.nn as nn
from data.ProxyTerm import ProxyTerm
from model.SparseLinear import SparseLinear


class Encoder(nn.Module):
    """Base class for fully connected dense encoder."""

    def __init__(self, go_layers, dtype):
        super(Encoder, self).__init__()

        # Initialize architecture
        self.go_layers = list(reversed(go_layers))
        network_layers = []
        self.activation = nn.ReLU()
        for i in range(len(self.go_layers) - 1):
            network_layers.append(nn.Linear(len(self.go_layers[i]), len(self.go_layers[i + 1]), dtype=dtype))
            if i < len(self.go_layers) - 2:
                network_layers.append(self.activation)
            # Final layer has no activation
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
        """Implemented in BIEncoder subclass."""
        return

    def mask_gradients(self):
        """Implemented in BIEncoder subclass."""
        return


class BIEncoder(Encoder):
    """Encoder subclass. Uses dense weights and dense masks derived from provided GO structure to construct a biologically-informed dense encoder."""

    def __init__(self, go_layers, dtype, masks=None):
        super(BIEncoder, self).__init__(go_layers, dtype)

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
                child = current_layer[i]
                parent_ids = [parent.item_id for parent in child.parents]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in parent_ids:
                        edge_mask[j][i] = 1
            edge_masks.append(edge_mask)
        return edge_masks


class SparseEncoder(nn.Module):
    """Encoder class. Uses sparse weights and sparse masks derived from provided GO structure (or from file) to construct a biologically-informed sparse encoder."""

    def __init__(self, go_layers, dtype, masks=None, protocol="coo"):
        super(SparseEncoder, self).__init__()
        self.go_layers = list(reversed(go_layers))
        self.protocol = protocol

        # Initialize masks
        if masks:
            self.edge_masks = masks[0]
            self.proxy_masks = masks[1]
        else:
            self.edge_masks = self._create_edge_masks()
            self.proxy_masks = self._create_proxy_masks()

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
        # Apply proxy mask to force weights to 1
        self.mask_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def mask_weights(self):
        """Sparse weight matrices ensure that edgeless weights remain zero. Dense proxy masks are used to set the non-zero weights corresponding to a ProxyTerm to 1, and their bias to 0."""
        mask_index = 0
        for layer in self.layers:
            if isinstance(layer, SparseLinear):
                # Ensure that devices match
                if self.proxy_masks[mask_index].device != layer.weight.device:
                    self.proxy_masks[mask_index] = self.proxy_masks[mask_index].to(layer.weight.device)

                nnz_rows = layer.weight.data.coalesce().indices()[0]
                proxy_mask = self.proxy_masks[mask_index]
                # If a row of the sparse weight matrix corresponds to a ProxyTerm, all non-zero values in that row are set to 1
                for j in range(len(nnz_rows)):
                    if proxy_mask[nnz_rows[j]]:
                        layer.weight.data = layer.weight.data.coalesce()
                        layer.weight.data.values()[j] = 1
                # Mask ProxyTerm bias
                layer.bias.data = torch.masked_fill(layer.bias.data, proxy_mask.T, value=0)
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
        """Returns a list of sparse 2D boolean tensors. Each tensor functions as an adjacency matrix between network layers, derived from GO."""
        edge_masks = []
        for n in range(len(self.go_layers) - 1):
            current_layer = self.go_layers[n]
            next_layer = self.go_layers[n + 1]
            non_zero_indices = []
            # For each layer, find the coordinates of the non-zero values of the adjacency matrix
            for i in range(len(current_layer)):
                child = current_layer[i]
                parent_ids = [parent.item_id for parent in child.parents]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in parent_ids:
                        non_zero_indices.append([j, i])

            # Construct a sparse boolean mask using the non-zero coordinates
            edge_mask = torch.sparse_coo_tensor(torch.tensor(non_zero_indices).t(), torch.ones(len(non_zero_indices)),
                                                size=(len(next_layer), len(current_layer)), dtype=torch.bool)
            edge_masks.append(edge_mask.coalesce())
        return edge_masks
