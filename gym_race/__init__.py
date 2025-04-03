from gymnasium.envs.registration import register

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)

# Register enhanced version with more sensors and actions
register(
    id='Pyrace-v3',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
