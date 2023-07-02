import argparse

import configs
import trainers


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--config_id",
        type=str,
        choices=["podcnn_trainer", "lego3d_config", "cmamae_config"],
        required=True,
    )

    return argument_parser.parse_args()


def run(config_id):
    valid_ids = list(trainers.TRAINERS_MAP.keys())
    if config_id not in valid_ids:
        err_msg = f"{config_id} key expected in {valid_ids}"
        raise KeyError(err_msg)

    trainer = trainers.TRAINERS_MAP[config_id](config_id)
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    run(args.config_id)
