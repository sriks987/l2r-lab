"""Simple Random Agent."""
import json
import time
import numpy as np
from src.agents.base import BaseAgent
from src.utils.utils import ActionSample
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath

@yamlize
class RandomAgent(BaseAgent):
    """Randomly pick actions in the space."""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        polyak: float,
        lr: float,
        actor_critic_cfg_path: str,
        load_checkpoint_from: str = "",
    ):
        super(RandomAgent, self).__init__()


    def select_action(self, obs) -> np.array:
        """Selection action through random sampling.

        Args:
            obs (np.array): Observation (unused)

        Returns:
            np.array: Action
        """
        action_obj = ActionSample()
        action_obj.action = self.action_space.sample()

        return action_obj
    
    def register_reset(self, obs) -> np.array:
        """Handle reset of episode.

        Args:
            obs (np.array): Observation

        Returns:
            np.array: Action
        """
        pass

    def update(self, data):
        """Model update given data

        Args:
            data (dict): Data.
        """

        pass

    def load_model(self, path):
        """Load model checkpoint from path

        Args:
            path (str): Path to checkpoint
        """

        # Nothing to load since random agent
        pass

    def save_model(self, path):
        """Save model checkpoint to path

        Args:
            path (str): Path to checkpoint
        """

        # Nothing to save since random agent
        pass