import torch
import torch.nn as nn
from data.ProxyTerm import ProxyTerm


class BIEncoder(nn.Module):
    def __init__(self, go_layers):
        super(BIEncoder, self).__init__()
        go_layers = list(reversed(go_layers))

        # Initialize biologically-informed masks
        self.proxy_masks = self._create_proxy_masks(go_layers)
        self.edge_masks = self._create_edge_masks(go_layers)

        # Initialize architecture
        network_layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            network_layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            if i < len(go_layers) - 2:
                network_layers.append(self.relu)
        self.layers = nn.ModuleList(network_layers)

        # Initialize all weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        # Mask weights using proxy and edge masks
        self.mask_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def mask_weights(self):
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

    @staticmethod
    def _create_proxy_masks(go_layers):
        proxy_masks = []
        for n in range(len(go_layers) - 1):
            next_layer = go_layers[n + 1]
            proxy_mask = torch.zeros(len(next_layer), 1, dtype=torch.bool)
            for i in range(len(next_layer)):
                if isinstance(next_layer[i], ProxyTerm):
                    proxy_mask[i] = 1
            proxy_masks.append(proxy_mask)
        return proxy_masks

    @staticmethod
    def _create_edge_masks(go_layers):
        edge_masks = []
        for n in range(len(go_layers) - 1):
            current_layer = go_layers[n]
            next_layer = go_layers[n + 1]
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


class Decoder(nn.Module):
    def __init__(self, go_layers):
        super(Decoder, self).__init__()
        self.layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            self.layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            if i < len(go_layers) - 2:
                self.layers.append(self.relu)
            else:
                self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z):
        return self.layers(z)


class Autoencoder(nn.Module):
    def __init__(self, go_layers):
        super(Autoencoder, self).__init__()
        self.encoder = BIEncoder(go_layers)
        self.decoder = Decoder(go_layers)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
