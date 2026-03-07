# BARK custom Gymnasium environments
from envs.bark_ant_3leg import BarkAnt3LegEnv, register_bark_envs
from envs.bark_go1_3leg import BarkGo1_3LegEnv, register_bark_go1_envs

register_bark_envs()
register_bark_go1_envs()

__all__ = [
    "BarkAnt3LegEnv",
    "BarkGo1_3LegEnv",
    "register_bark_envs",
    "register_bark_go1_envs",
]
