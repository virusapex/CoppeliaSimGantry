from gymnasium.envs.registration import register


register(
    id='Gantry-v0',
    entry_point='Gantry.envs:GantryEnv',
    max_episode_steps=100,
)

register(
    id='Gantry-v1',
    entry_point='Gantry.envs.GantryEnv_v1:GantryEnv',
    max_episode_steps=100,
)

register(
    id='Gantry-v1_cnn',
    entry_point='Gantry.envs.GantryEnv_v1_cnn:GantryEnv',
    max_episode_steps=100,
)
