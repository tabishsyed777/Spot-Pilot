from gymnasium.envs.registration import register
from .env import SpotEnv
register(id='SpotPilot-v0',entry_point='spotpilot:SpotEnv')
__all__=['SpotEnv']
