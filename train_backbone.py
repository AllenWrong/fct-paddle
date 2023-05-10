from argparse import ArgumentParser

import paddle
import yaml
from utils.getters import get_model, get_optimizer
import paddle.distributed as dist
from trainers import BackboneTrainer
from dataset.SubImageFolder import get_image_folder
from utils.scheduler import get_policy
from paddle import nn
from utils.net_utils import LabelSmoothing, backbone_to_paddlescript
import os
import init
import numpy as np
import random
import pickle


def load_state(pd_model, state_name):
    with open(state_name, "rb") as f:
        state = pickle.load(f)
    pd_model.load_dict(state)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    model = get_model(config.get('arch_params'))
    # load_state(model, "resnet50")
    # init.reset_initialized_parameter(model)

    if dist.is_initialized():
        model = paddle.DataParallel(model)

    trainer = BackboneTrainer()
    optimizer = get_optimizer(model, **config.get("optimizer_params"))
    data = get_image_folder(**config.get("dataset_params"))

    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))

    if config.get('label_smoothing') is None:
        criterion = nn.CrossEntropyLoss()
    else:

        criterion = LabelSmoothing(smoothing=config.get('label_smoothing'))

    for epoch in range(config['epochs']):
        lr_policy(epoch, iteration=None)

        train_acc1, train_acc5, train_loss = trainer.train(
            train_loader=data.train_loader, model=model, criterion=criterion, optimizer=optimizer
        )

        train_info = f"Train: epoch = {epoch}, Loss = {train_loss}, Top 1 = {train_acc1}, Top 5 = {train_acc5}, lr = {optimizer.get_lr()}\n"
        print(train_info, end="")

        log_path = os.path.join(args.log_dir, "./train_log.log")
        with open(log_path, "a") as f:
            f.write(train_info)


        test_acc1, test_acc5, test_loss = trainer.valid(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion
        )

        test_info = f"Test: epoch = {epoch}, Loss = {test_loss}, Top 1 = {test_acc1}, Top 5 = {test_acc5}\n"
        print(test_info, end="")
        with open(log_path, "a") as f:
            f.write(test_info)

    backbone_to_paddlescript(model, config['output_model_path'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    parser.add_argument('--log_dir', type=str, default="/root/paddlejob/workspace/output/")

    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
        
    paddle.seed(5)
    # dist prepare
    dist.init_parallel_env()
    main(read_config)
