import os 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# local imports
from .base_model import BaseModel
from utils import utils as ut 
from gym_pcgrl.envs.pcgrl_env_3d_piecewise import LegoPCGEnv3DPiecewise


class LegoModelPiecewise(BaseModel):

    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        # unpack config
        self.train_config = self.config.train
        self.model_config = self.config.model
        self.policy = self.train_config["policy"]
        self.num_timesteps = self.train_config["num_episodes"]*self.train_config["reps_per_episode"]*self.train_config["num_of_blocks"]
        self.lego_blocks_dims_dict = self.train_config["LegoBlockDimensions"]
        
        if not os.path.exists(self.model_config["log_path"]):
            os.mkdir(self.model_config["log_path"])
    
        savedir =  self.model_config["log_path"] + self.train_config["reward_param"] +"_"
        
        if self.train_config["scheduled_episodes"]:
            savedir += "sched_"
        
        else:
            savedir += str(self.train_config["reps_per_episode"]) + "_passes_"

        savedir +=  str(self.num_timesteps) + "_ts_" + str(self.train_config["num_of_blocks"]) + "_blocks_"  + str(self.train_config["observation_size"]) + "_obs"
        
        if self.train_config["punish"]:
            savedir += "_punish"

        iter = 0
        while os.path.exists(savedir +"_" + str(iter)):
            iter += 1

        self.animations_path = savedir +"_" + str(iter) + "/"
        os.mkdir(self.animations_path)

        self.log_path = self.animations_path + "/logdir"
        os.mkdir(self.log_path)

        self.saved_model_path = self.animations_path +"/model"

        self.model = None
        self.device = ut.get_device()
        self.env = DummyVecEnv([lambda: LegoPCGEnv3DPiecewise(self.train_config, self.animations_path)])

        #check_env(self.env)

        # train or load a trained model
        if mode == "train":
            self.build()
        else:
            self.load_model()
    
    def load_data(self):
        """
            Not relevant for RL models, suggest a name change
        """
        pass


    def build(self):
        self.model = PPO(self.policy, 
                        self.env, 
                        device=self.device, 
                        n_steps = min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                        batch_size =  min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                        tensorboard_log=self.log_path)

    def load_model(self):

        self.model = PPO.load(self.saved_model_path)

    def train(self):
        # will be moved to executor once callbacks are in place 
        self.model.learn(self.num_timesteps, reset_num_timesteps=False)
        self.model.save(self.saved_model_path)

        self.evaluate()

    def evaluate(self):
        # will be moved to evaluator later 
        #self.model = self.load_model()
        curr_obs = self.env.reset()
        curr_step_num = 0 

        ut.save_arrangement(
                self.env.envs[0].rep.blocks, 
                self.animations_path, 
                curr_step_num, 
                self.env.envs[0].reward_history[-1], 
                render = True)

        while True:

            action, _ = self.model.predict(curr_obs)
            # print(action)
            curr_obs, _, is_finished, info = self.env.step(action) 

            curr_step_num += 1

            ut.save_arrangement(
                self.env.envs[0].rep.blocks, 
                self.animations_path, 
                curr_step_num, 
                None, 
                self.env.envs[0].reward_history,
                render = True)
            

            if is_finished[0]:
                # curr_obs = env.reset()
                break
            elif curr_step_num > 1000:
                print("Long loop")
                # env.envs[0]._rep.final_map = np.copy(env.envs[0]._rep._map)
                break

        
        ut.animate(self.animations_path)

        
        print(self.env.envs[0].reward_history)