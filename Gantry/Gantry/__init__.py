from gym.envs.registration import register


register(
    id='Gantry-v0',
    entry_point='Gantry.envs:GantryEnv',
    max_episode_steps=250,
)
