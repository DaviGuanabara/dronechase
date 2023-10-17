from gymnasium.envs.registration import register


register(
    id="MyFirstEnv-v0",
    entry_point="loyal_wingmen.envs:MyFirstEnv",
)
