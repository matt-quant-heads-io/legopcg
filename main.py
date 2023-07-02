import argparse

import configs
import models as models


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--mode", choices=["inference", "train"], required=False, default="train"
    )
    argument_parser.add_argument(
        "--config_id",
        type=str,
        choices=["podcnn_config", "lego3d_config", "cmamae_config"],
        required=False,
        default="lego3d_config",
    )
    argument_parser.add_argument(
        "--model_id",
        type=str,
        choices=["podcnn_model", "lego3d_model", "cmamae_model"],
        required=False,
        default="lego3d_model",
    )
    argument_parser.add_argument(
        "--gen_train_data", required=False, action="store_true"
    )

    return argument_parser.parse_args()


def run(mode, config_id, model_id, gen_train_data):
    config = configs.CONFIGS_MAP[config_id]
    model = models.MODELS_MAP[model_id](config, mode)
    if mode == "train":
        if gen_train_data:
            model.generate_data()
        model.load_data()
        model.train()
    else:
        # model.run_inference()
        model.evaluate()


if __name__ == "__main__":
    args = get_args()
    run(args.mode, args.config_id, args.model_id, args.gen_train_data)
