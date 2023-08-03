import argparse

import configs
import model as models




def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--mode', choices=['inference', 'train'], required=True)
    argument_parser.add_argument('--config_id', type=str, choices=['podcnn_config', 'lego3d_config', 'cmamae_config', 'piecewise_config'], required=True)
    argument_parser.add_argument('--model_id', type=str, choices=['podcnn_model', 'lego3d_model', 'cmamae_model', 'piecewise_model'], required=True)
    argument_parser.add_argument('--gen_train_data', required=False, action="store_true")
    argument_parser.add_argument('--num_blocks', type = int, required=False)
    argument_parser.add_argument('--reps_per_ep', type = int, required=False)
    argument_parser.add_argument('--observation_size', type = int, required=False)
    argument_parser.add_argument('--punish', type = float, required = False)
    argument_parser.add_argument('--reward_param', type = str, required = False)
    argument_parser.add_argument('--controllable', required = False, action='store_true')

    return argument_parser.parse_args()


def run(mode, config_id, model_id, gen_train_data, num_blocks = None, reps_per_ep = None, observation_size = None, punish = None, reward_param = None, controllable=False):
    config = configs.CONFIGS_MAP[config_id]

    if num_blocks != None:
        config['train']['num_of_blocks'] = num_blocks
    if reps_per_ep != None:
        config['train']["reps_per_episode"] = reps_per_ep
    if observation_size != None:
        config['train']["observation_size"] = observation_size
    if punish != None:
        config['train']['punish'] = punish
    if reward_param != None:
        config['train']['reward_param'] = reward_param
    if controllable:
        config['train']['controllable'] = True

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
        args.gen_train_data,
        args.num_blocks,
        args.reps_per_ep,
        args.observation_size,
        args.punish,
        args.reward_param,
        args.controllable
    )