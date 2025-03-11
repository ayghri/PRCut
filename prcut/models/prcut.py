from torch import nn


def get_prcut_encoder(input_dim, num_clusters, num_layers=1, hidden_units=1024):
    assert num_layers > 0
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, num_clusters))
    else:

        layers.extend(
            [
                nn.Linear(input_dim, hidden_units),
                nn.GELU(),
            ]
        )

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_units, num_clusters))
    layers.append(nn.Softmax(-1))

    return nn.Sequential(*layers)
