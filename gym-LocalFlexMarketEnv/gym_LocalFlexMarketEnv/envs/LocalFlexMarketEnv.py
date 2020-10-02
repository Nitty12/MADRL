import gym
from gym import spaces
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import util
import pickle


class LocalFlexMarketEnv(gym.Env):
    """A local flexibility market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, SpotMarket, DSO, alg):
        super().__init__()
        self.SpotMarket = SpotMarket
        self.DSO = DSO
        self.alg = alg
        self.nAgents = len(self.SpotMarket.participants)
        self.agents = self.SpotMarket.participants
        self.time = 0
        self.startDay = 0
        self.endDay = self.SpotMarket.spotTimePeriod/self.SpotMarket.dailySpotTime
        self.checkTime = False
        """If needed change the limit of rewards here:"""
        self.reward_range = (-float('inf'), float('inf'))

        # to be used to alternate between spot and flex states
        self.spotState = True

        """ action is the hourly strategic bid multiplier(sbm) and price multiplier (spm) - float values
            the bid qty is given by sbm*maxPower
                if sbm = 0, the bidder does not bid
                if sbm = 1, bidder bids maximum power
            the bid price is given by spm*marginalCost
                if spm = 1, the bidder reveals its true cost
                if spm >1, bidder is trying to increase profit
            action space is the same for spot and flex market states even though we dont consider spm in spot 
            since there cant be 2 action space spec.
            Currently, the action space has sbm_spot - 24, sbm_flex - 24, spm - 24
        """
        self.action_space = []
        self.total_qmix_action_space = []

        """ observation is 
                hourly market clearing prices of 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Flex market
                spot or flex status
        """
        self.observation_space = []
        """since qmix uses DQN, the action space is (1,) for all agents, thus a total action space of (72,)
            is divided as ((1,),(1,),...) for the total number of agents.
            eg., 5 agents -> total_qmix_observation_space: ((1,),(1,),...360 times) """
        self.total_qmix_observation_space = []
        
        for agent in self.agents:
            total_action_space = [] 
            minLimit, maxLimit = agent.getActionLimits(self.alg)
            if self.alg == 'QMIX' or self.alg == 'IQL':
                agent_action_space = spaces.Box(low=minLimit, high=maxLimit, dtype=int)
                agent_individual_action_space = []
                for i in range(len(minLimit)):
                    agent_individual_action_space.append(spaces.Box(low=minLimit[i], high=maxLimit[i],
                                                                    shape=(1,), dtype=int))
                self.total_qmix_action_space.extend(agent_individual_action_space)
                agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf,
                                                     shape=(3 * self.SpotMarket.dailySpotTime + 1,), dtype=np.float32)
                """this reduced obs (providing only the MCP of current hour) will be given to the Q network for each hour"""
                self.total_qmix_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf,
                                                     shape=(2 * self.SpotMarket.dailySpotTime + 2,), dtype=np.float32))
            elif self.alg == 'MADDPG':
                agent_action_space = spaces.Box(low=minLimit, high=maxLimit, dtype=np.float32)
                agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf,
                                                     shape=(3 * self.SpotMarket.dailySpotTime + 1,), dtype=np.float32)
            total_action_space.append(agent_action_space)
            if len(total_action_space) > 1:
                self.action_space.append(spaces.Tuple(total_action_space))
            else:
                self.action_space.append(total_action_space[0])
            self.observation_space.append(agent_observation_space)

        """Convert to tuple for compatibility with tf-agents"""
        self.action_space = spaces.Tuple(self.action_space)
        self.total_qmix_action_space = spaces.Tuple(self.total_qmix_action_space)
        self.observation_space = spaces.Tuple(self.observation_space)
        self.total_qmix_observation_space = spaces.Tuple(self.total_qmix_observation_space)

    def step(self, action):
        obs = []
        reward = []
        done = []
        info = {'x': []}
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action[i], agent, self.action_space[i], self.spotState)
        if self.spotState:
            self.spotStep()
        else:
            self.flexStep()

        # record observation for each agent
        for agent in self.agents:
            obs.append(self._get_obs(agent))
            reward.append(self._get_reward(agent))
            done.append(self._get_done(agent))
            info['x'].append(self._get_info(agent))

        self.time += 1
        self.spotState = not self.spotState

        if done[0]:
            self.reset()

        return tuple(obs), tuple(reward), done, info

    def reset(self):
        self.time = 0
        self.spotState = True

        # reset the environment
        self.SpotMarket.reset(self.startDay)
        self.DSO.reset()
        for agent in self.agents:
            agent.reset(self.startDay)

        # record observations for each agent
        obs = []
        for agent in self.agents:
            obs.append(self._get_obs(agent))

        return tuple(obs)

    # get info used for evaluation
    def _get_info(self, agent):
        return {}

    # get observation for a particular agent
    def _get_obs(self, agent):
        MCP, spotDispatch, flexDispatch, spotState = agent.getObservation(self.startDay)
        return np.hstack((MCP, spotDispatch, flexDispatch, spotState))

    # get dones for a particular agent
    def _get_done(self, agent):
        """when to reset the environment?
        currently after 1 year"""
        if self.SpotMarket.day + 1 >= self.endDay:
            return True
        else:
            return False

    # get reward for a particular agent
    def _get_reward(self, agent):
        spotReward, flexReward = agent.getReward()
        if self.spotState:
            return spotReward
        else:
            return flexReward


    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, spotState):
        """ action is the hourly strategic bid multiplier(sbm) and price multiplier (spm) for the agent
        """
        if spotState:
            """action is a rank 2 tensor [[]] with just one row, so we take action[0][indexing]"""
            agent.spotBidMultiplier = action[0][:self.SpotMarket.dailySpotTime]
            # we dont use the price multiplier in spot market

        else:
            agent.flexBidMultiplier = action[0][self.SpotMarket.dailySpotTime:2*self.SpotMarket.dailySpotTime]
            agent.flexBidPriceMultiplier = action[0][2*self.SpotMarket.dailySpotTime:]

    def spotStep(self):
        lastTime = time.time()
        self.SpotMarket.collectBids()
        self.SpotMarket.sendDispatch()
        if self.checkTime:
            util.checkTime(lastTime, 'Spot step')

    def flexStep(self):
        lastTime = []
        lastTime.append(time.time())
        if self.DSO.checkCongestion(self.SpotMarket.bidsDF):
            if self.checkTime:
                util.checkTime(lastTime[-1], 'Power flow approximation')
                lastTime.append(time.time())
            self.DSO.askFlexibility()
            if self.checkTime:
                util.checkTime(lastTime[-1], 'Getting flex bids')
                lastTime.append(time.time())
            self.DSO.optFlex()
            if self.checkTime:
                util.checkTime(lastTime[-1], 'choosing the flexibility')
        self.DSO.endDay()
        self.SpotMarket.endDay()
        if self.checkTime:
            util.checkTime(lastTime[0], 'Flex step total')

    def render(self, mode='human'):
        if self.time % 20 == 0:
            for agent in self.agents:
                print("Agent ID: {} Reward: {}".format(agent.id, agent.getReward()))
