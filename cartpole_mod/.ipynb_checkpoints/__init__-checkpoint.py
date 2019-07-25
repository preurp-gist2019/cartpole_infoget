from gym.envs.registration import register

register(id='CartPole-v3', entry_point='cartpole_mod.envs:CartPoleEnv')
