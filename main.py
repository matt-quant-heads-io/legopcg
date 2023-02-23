import argparse

import configs
import model as models


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--mode', choices=['inference', 'train'], required=True)
    argument_parser.add_argument('--config_id', type=str, choices=['podcnn_config', 'lego3d_config'], required=True)
    argument_parser.add_argument('--model_id', type=str, choices=['podcnn_model', 'lego3d_model'], required=True)
    argument_parser.add_argument('--gen_train_data', required=False, action="store_true")

    return argument_parser.parse_args()


def run(mode, config_id, model_id, gen_train_data):
    config = configs.CONFIGS_MAP[config_id]
    model = models.MODELS_MAP[model_id](config)
    if mode == 'train':
        if gen_train_data:
            model.generate_data()
        model.load_data()
        model.train()
    else:
        model.run_inference()


if __name__ == "__main__":
    args = get_args()
    run(
        args.mode,
        args.config_id,
        args.model_id,
        args.gen_train_data
    )