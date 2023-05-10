from argparse import ArgumentParser

import yaml
import paddle
from dataset.SubImageFolder import SubImageFolder
from utils.eval_utils import cmc_evaluate
import os


def main(config):
    gallery_model = paddle.jit.load(config.get('gallery_model_path'))
    query_model = paddle.jit.load(config.get('query_model_path'))

    data = SubImageFolder(**config.get('dataset_params'))

    cmc_out, mean_ap_out = cmc_evaluate(
        gallery_model,
        query_model,
        data.val_loader,
        **config.get('eval_params')
    )

    if config.get('eval_params').get('per_class'):
        
        info =  'CMC Top-1 = {}, CMC Top-5 = {}\n'.format(*cmc_out[0])
        info += 'Per class CMC: {}'.format(cmc_out[1])
    else:
        info = 'CMC Top-1 = {}, CMC Top-5 = {}\n'.format(*cmc_out)

    if mean_ap_out is not None:
        info += 'mAP = {}\n'.format(mean_ap_out)

    print(info)
    with open(os.path.join(args.log_dir, "eval.log"), "a") as f:
        f.write(args.config + "\n")
        f.write(info)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    parser.add_argument('--log_dir', type=str, default="/root/paddlejob/workspace/output/",
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
