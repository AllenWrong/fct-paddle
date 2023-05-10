from typing import Optional

import paddle
from paddle import nn


class ConvBlock(nn.Layer):
    """Convenience convolution module."""

    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 normalizer: Optional[nn.Layer] = nn.BatchNorm2D,
                 activation: Optional[nn.Layer] = nn.ReLU) -> None:
        """Construct a ConvBlock module."""
        super().__init__()

        self.conv = nn.Conv2D(
            channels_in, channels_out,
            kernel_size=kernel_size, stride=stride,
            bias_attr=normalizer is None,
            padding=kernel_size // 2
        )
        if normalizer is not None:
            self.normalizer = normalizer(channels_out)
        else:
            self.normalizer = None
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        x = self.conv(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLP_BN_SIDE_PROJECTION(nn.Layer):
    """FCT transformation module."""

    def __init__(self,
                 old_embedding_dim: int,
                 new_embedding_dim: int,
                 side_info_dim: int,
                 inner_dim: int = 2048,
                 **kwargs) -> None:
        """Construct MLP_BN_SIDE_PROJECTION module.
        """
        super().__init__()

        self.inner_dim = inner_dim
        self.p1 = nn.Sequential(
            ConvBlock(old_embedding_dim, 2 * old_embedding_dim),
            ConvBlock(2 * old_embedding_dim, 2 * new_embedding_dim),
        )

        self.p2 = nn.Sequential(
            ConvBlock(side_info_dim, 2 * side_info_dim),
            ConvBlock(2 * side_info_dim, 2 * new_embedding_dim),
        )

        self.mixer = nn.Sequential(
            ConvBlock(4 * new_embedding_dim, self.inner_dim),
            ConvBlock(self.inner_dim, self.inner_dim),
            ConvBlock(self.inner_dim, new_embedding_dim, normalizer=None,
                      activation=None)
        )

    def forward(self,
                old_feature: paddle.Tensor,
                side_info: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass.
        """
        x1 = self.p1(old_feature)
        x2 = self.p2(side_info)
        return self.mixer(paddle.concat([x1, x2], axis=1))
