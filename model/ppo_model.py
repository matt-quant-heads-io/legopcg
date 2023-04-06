from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv


from utils import utils as ut 
from .base_model import BaseModel

class PPOModel(BaseModel):
    """
        Base PPO model with code which is standard to 
        any PPO based model. 
    """

    def __init__(self, cfg):
        """
            Set up standard variables
        """
        super().__init__(cfg)

        # unpack config
        self.train_config = self.config.train
        self.model_config = self.config.model
        self.policy = self.train_config["policy"]
        self.num_timesteps = self.train_config["num_timesteps"]
        self.num_envs = self.train_config["num_envs"]
        self.log_path = self.model_config["log_path"]
        self.saved_model_path = f'{self.model_config["saved_model_path"]}/{self.model_config["model_name"]}'
        self.animations_path = self.model_config["animations_path"]

        self.model = None
        self.env = None
        self.device = ut.get_device()


    def get_vector_env(self, func):
        """
            Return a vector of parallel environments 
            Passed func enclose all the wrappers a user want
        """
        return DummyVecEnv([func() for _ in range(self.num_envs)])
    
    def load_data(self):
        """
            Not relevant for RL models, suggest a name change
        """
        pass

    def build(self):
        """
            Standard way of building a PPO Model
        """

        assert self.env is not None, "Call 'get_vector_env' method first to make the env"

        policy_kwargs = dict(net_arch=[128, 128])

        self.model = PPO(self.policy, 
                        self.env, 
                        device=self.device, 
                        policy_kwargs=policy_kwargs)
                        # tensorboard_log=self.log_path)
        
    def load_model(self):

        self.model = PPO.load(self.saved_model_path)
    
    def train(self):
        """
            Standard train procedure for a PPO model
        """
        
        assert self.model is not None, "Call 'build' method first to make the model"

        self.model.learn(total_timesteps=self.num_timesteps)
        self.model.save(self.saved_model_path)
        
    
    def evaluate(self):
        """
            Implement model specific evaluate
        """
        raise NotImplementedError