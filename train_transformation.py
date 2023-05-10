import os
from argparse import ArgumentParser
import yaml
from PIL import ImageFile
import paddle

ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.SubImageFolder import get_image_folder
from utils.net_utils import transformation_to_paddlescripts
from utils.scheduler import get_policy
from utils.getters import get_model, get_optimizer
import paddle.distributed as dist
from trainers import TransformationTrainer
from paddle import nn


def main(config):
    model = get_model(config.get("arch_params"))
    old_model = paddle.jit.load(config.get('old_model_path'))
    new_model = paddle.jit.load(config.get('new_model_path'))

    if dist.is_initialized():
        model = paddle.DataParallel(model)
        old_model = paddle.DataParallel(old_model)
        new_model = paddle.DataParallel(new_model)

    if config.get('side_info_model_path') is not None:
        side_info_model = paddle.jit.load(config.get('side_info_model_path'))
        if dist.is_initialized():
            side_info_model = paddle.DataParallel(side_info_model)
    else:
        side_info_model = old_model

    trainer = TransformationTrainer(old_model, new_model, side_info_model)

    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    print("lr=", optimizer.get_lr())
    data = get_image_folder(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))

    criterion = nn.MSELoss()
    for epoch in range(config.get("epochs")):
        lr_policy(epoch, iteration=None)

        if config.get('switch_mode_to_eval'):
            switch_mode_to_eval = epoch >= config.get('epochs') / 2
        else:
            switch_mode_to_eval = False

        train_loss = trainer.train(
            train_loader=data.train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            switch_mode_to_eval=switch_mode_to_eval,
        )

        train_info = f"Train: epoch = {epoch}, Average Loss = {train_loss}, lr = {optimizer.get_lr()}\n"

        print(train_info, end="")

        log_path = os.path.join(args.log_dir, "train_transformation.log")
        with open(log_path, "a") as f:
            f.write(train_info)

        # evaluate on validation set
        test_loss = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
        )

        test_info = f"Test: epoch = {epoch}, Average Loss = {test_loss}\n"
        print(test_info, end="")
        with open(log_path, "a") as f:
            f.write(test_info)

    transformation_to_paddlescripts(old_model, side_info_model, model,
                                   config.get('output_transformation_path'),
                                   config.get('output_transformed_old_model_path'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    parser.add_argument('--log_dir', type=str, default="/root/paddlejob/workspace/output/")

    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)

    dist.init_parallel_env()
    main(read_config)
