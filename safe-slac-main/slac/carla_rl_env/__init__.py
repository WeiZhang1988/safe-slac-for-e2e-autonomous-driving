from gym.envs.registration import register

register(
    id='CarlaRlEnv-v0',
    entry_point='slac.carla_rl_env.carla_rl_env:CarlaRlEnv',
)

