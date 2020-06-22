from gym.envs.registration import register

register(
    id='LocalFlexMarketEnv-v0',
    entry_point='gym_LocalFlexMarketEnv.envs:LocalFlexMarketEnv',
    kwargs={
        'SpotMarket': None,
        'DSO': None,
        'grid': None,
    },
)