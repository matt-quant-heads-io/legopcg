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
    argument_parser.add_argument('--model_name', type=str, required=False)

    return argument_parser.parse_args()


def run(mode, config_id, model_id, gen_train_data, num_blocks = None, reps_per_ep = None, observation_size = None, punish = None, reward_param = None, controllable=False, model_name=False):
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
    if model_name is not None:
        model_name_list = model_name.split("_")
        if model_name_list[0] == 'controllable':
            config['train']['controllable'] = True
            config['model']['features_extractor'] =  model_name_list[3]
            config['train']['action_space'] = model_name_list[4]+"_position"
            config['train']['reps_per_episode'] = int(model_name_list[6])
            config['train']['num_of_blocks'] = int(model_name_list[10])
            config['train']['cnn_output_channels'] = int(model_name_list[14])
            config['train']['punish'] = int(model_name_list[-2])
        else:
            config['train']['controllable'] = False
            config['model']['features_extractor'] =  model_name_list[2]
            config['train']['action_space'] = model_name_list[3]+"_position"
            config['train']['reps_per_episode'] = int(model_name_list[5])
            config['train']['num_of_blocks'] = int(model_name_list[9])
            config['train']['cnn_output_channels'] = int(model_name_list[13])
            config['train']['punish'] = int(model_name_list[-2])

#controllan_26_passes_5200000_ts_20_blocks_31_obs_48_chans_punish_0.035_1
    model = models.MODELS_MAP[model_id](config)

    if mode == 'train':
        if gen_train_data:
            model.generate_data()
        model.load_data()
        if model_name is not None:
            model_path = model.saved_model_path.split("/")
            model_path[-3] = model_name
            model.saved_model_path = "/".join(model_path)
            model.load()
        model.train()
    else:
        if model_name == None:
            print("Running inference requires argument passed for --model_name.")
            quit()
        model_path = model.saved_model_path.split("/")
        model_path[-3] = model_name
        model.saved_model_path = "/".join(model_path)
        model.load()
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