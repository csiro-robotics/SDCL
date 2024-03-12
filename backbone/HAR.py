from torch import nn, Tensor
import math

from backbone import MammothBackbone, xavier


def _mlp_layer(in_features: int, out_features: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_features, out_features), nn.Dropout(dropout), nn.ReLU()
    )


class SimpleMLP(MammothBackbone):
    """
    MLPEncoder implements a multi-layer perceptron encoder for an autoencoder.
    """

    def __init__(
        self,
        latent_size: int,
        data_shape: tuple,
        width: int,
        layer_count: int,
        dropout: float,
        n_classes: int,
        layer_growth: float = 2.0,
    ) -> None:
        """_summary_

        :param latent_size: The size of the latent representation.
        :param data_shape: The shape of the input data.
        :param width: The width of the hidden layers. Each hidden layer will have
        half the width of the previous layer.
        :param layer_count:  The number of hidden layers.
        :param dropout:  The dropout rate.
        """
        super().__init__()
        total_features = math.prod(data_shape)
        latent_size = int(latent_size)
        width = int(width)
        layer_count = int(layer_count)

        layers = [nn.Flatten()]
        features_in = total_features
        features_out = width
        for _ in range(layer_count - 1):
            layers.append(_mlp_layer(features_in, features_out, dropout))
            features_in = features_out
            features_out = int(features_out / layer_growth)

        layers.append(nn.Linear(features_in, latent_size))
        self.classifier = nn.Linear(latent_size, n_classes)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, returnt="out") -> Tensor:
        feats = self.layers(x)

        if returnt == "features":
            return feats

        out = self.classifier(feats)
        if returnt == "out":
            return out
        elif returnt == "all":
            return (out, feats)
        raise NotImplementedError("Unknown return type")

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)


def PAMAP2_mlp():
    return SimpleMLP(
        latent_size=64,
        data_shape=(243,),
        width=256,
        layer_count=5,
        dropout=0.1,
        n_classes=12,
        layer_growth=1.0,
    )


def DSADS_mlp():
    return SimpleMLP(
        latent_size=128,
        data_shape=(405,),
        width=512,
        layer_count=5,
        dropout=0.1,
        n_classes=18,
        layer_growth=1.0,
    )
