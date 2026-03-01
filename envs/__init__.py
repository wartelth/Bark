# BARK custom Gymnasium environments
from envs.bark_ant_3leg import BarkAnt3LegEnv, register_bark_envs

register_bark_envs()

__all__ = ["BarkAnt3LegEnv", "register_bark_envs"]
