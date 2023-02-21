import argparse

import configs
from model.pod_cnn import PoDCNN
from model.lego_model_3d import LegoModel3D

def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--mode', choices=['inference', 'train'], required=True)
    argument_parser.add_argument('--config_id', type=str, 
                choices=['podcnn_config', "lego3d_config"], required=True)
    argument_parser.add_argument('--gen_train_data', required=False, action="store_true")

    return argument_parser.parse_args()


def run(mode, config_id, gen_train_data):
    
    print("running main")

    config = configs.CONFIGS_MAP[config_id]

    # standardize this by creating a dict for models 
    # model = PoDCNN(config)
    model = LegoModel3D(config, mode)


    if mode == 'train':
        if gen_train_data:
            model.generate_data()
        model.load_data()
        model.train()
    else:
        # model.run_inference()
        model.evaluate()


if __name__ == "__main__":
    args = get_args()
    run(
        args.mode,
        args.config_id,
        args.gen_train_data
    )