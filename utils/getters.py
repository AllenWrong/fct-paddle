from typing import Dict, Optional

import paddle
import paddle.nn as nn
import models


def get_model(arch_params: Dict, **kwargs) -> nn.Layer:
    """Get a model given its configurations."""

    print("=> Creating model '{}'".format(arch_params.get('arch')))
    model = models.__dict__[arch_params.get('arch')](**arch_params)
    return model


def get_optimizer(model: nn.Layer,
                  algorithm: str,
                  lr,
                  weight_decay: float,
                  momentum: Optional[float] = None,
                  no_bn_decay: bool = False,
                  nesterov: bool = False,
                  **kwargs) -> paddle.optimizer.Optimizer:
    """Get an optimizer given its configurations."""

    if algorithm == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if
                     ("bn" in n) and not v.stop_gradient]
        rest_params = [v for n, v in parameters if
                       ("bn" not in n) and not v.stop_gradient]

        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr,
            parameters=[
                {
                    "params": bn_params,
                    "weight_decay": 0 if no_bn_decay else weight_decay,
                },
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            momentum=momentum,
            weight_decay=weight_decay,
            use_nesterov=nesterov,
        )
    elif algorithm == "adam":
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            parameters=filter(lambda p: not p.stop_gradient, model.parameters()),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"unsupported optimizer {algorithm}!")

    return optimizer
