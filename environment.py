import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers

tf.compat.v1.enable_v2_behavior()


class flexMarketEnv(py_environment.PyEnvironment):

    def __init__(self, periodPerDay=24):
        """ action is the hourly strategic bidding multiplier(sbm)
            so, the bid price is given by sbm*marginalCost
            if sbm = 1, the bidder reveals its true cost
            if sbm >1, bidder is trying to increase profit
        """
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(periodPerDay,), dtype=np.float32, minimum=1, maximum=5, name='action')

        """ observation is 
                hourly market clearing prices of 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Day ahead market,
                hourly load forecast of 't+1' day
        """
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(72,), dtype=np.float32, name='observation')

        """ initial state of the agent
        """
        self._state = np.zeros((72,), dtype=np.float32)

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((72,), dtype=np.float32)
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.flexMarketClearing(action)

        if self.simulationOver():
            self._episode_ended = True

        # TODO
        reward = 0.0

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=0.0, discount=0.9)

    def flexMarketClearing(self, action):
        # TODO
        pass

    def simulationOver(self):
        # some condition to end the simulation
        # TODO
        pass


if __name__ == '__main__':
    env = flexMarketEnv()
    utils.validate_py_environment(env, episodes=5)

    tl_env = wrappers.TimeLimit(env, duration=50)

    time_step = tl_env.reset()
    print(time_step)
    rewards = time_step.reward
