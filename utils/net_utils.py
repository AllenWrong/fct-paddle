import paddle
from paddle import nn


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


class LabelSmoothing(nn.Layer):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing: float = 0.0):
        """Construct LabelSmoothing module."""
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass.
        Args:
            x: Logits tensor.
            target: Ground truth target classes.

        Returns:Loss tensor.
        """
        logprobs = paddle.nn.functional.log_softmax(x, axis=-1)

        nll_loss = - paddle_gather(logprobs, dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FeatureExtractor(nn.Layer):
    """A wrapper class to return only features (no logits)."""

    def __init__(self, model):
        """Construct FeatureExtractor module."""
        super().__init__()
        self.model = model

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass.
        :param x: Input data.
        :return: Feature tensor computed for x.
        """
        _, feature = self.model(x)
        return feature


class TransformedOldModel(nn.Layer):
    """A wrapper class to return transformed features."""

    def __init__(self,
                 old_model,
                 side_model,
                 transformation) -> None:
        """Construct TransformedOldModel module."""
        super().__init__()
        self.old_model = old_model
        self.transformation = transformation
        self.side_info_model = side_model

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        old_feature = self.old_model(x)
        side_info = self.side_info_model(x)
        recycled_feature = self.transformation(old_feature, side_info)
        return recycled_feature


def prepare_model_for_export(model):
    if isinstance(model, paddle.DataParallel):
        model = model._layers

    model.eval()
    model.to("cpu")
    return model


def backbone_to_paddlescript(model, output_model_path):
    model = prepare_model_for_export(model)
    f = FeatureExtractor(model)
    model_script = paddle.jit.to_static(
        f, 
        input_spec=[paddle.static.InputSpec(shape=[-1, 3, 224, 224])]
    )
    paddle.jit.save(model_script, output_model_path)


def transformation_to_paddlescripts(
        old_model, side_model, transformation,
        output_transformation_path: str,
        output_transformed_old_model_path: str) -> None:
    """Convert a transformation model to torchscript."""
    transformation = prepare_model_for_export(transformation)
    old_model = prepare_model_for_export(old_model)
    side_model = prepare_model_for_export(side_model)

    model_script = paddle.jit.to_static(
        transformation,
        input_spec=[paddle.static.InputSpec(shape=[-1, 128, 1, 1]), paddle.static.InputSpec(shape=[-1, 128, 1, 1])]
    )
    paddle.jit.save(model_script, output_transformation_path)

    f = TransformedOldModel(old_model, side_model, transformation)
    model_script = paddle.jit.to_static(
        f,
        input_spec=[paddle.static.InputSpec(shape=[-1, 3, 224, 224])]
    )
    paddle.jit.save(model_script, output_transformed_old_model_path)