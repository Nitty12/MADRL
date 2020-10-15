from AgentNeuralNet import MADDPGAgent, QMIX, IQL
from SpotMarket import SpotMarket
from DSO import DSO
from Grid import Grid
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
import os
import time
import re
import util
import pickle
import tqdm
import optuna
import joblib
import datetime
import random
pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    st = time.time()
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    """specify the algorithm - MADDPG, QMIX, IQL
       specify the start and end days of the usecases:
            days 330-360: high overload possibility
            days 105-135 or 300-330: low overload possibility"""
    alg = 'MADDPG'
    startDay = 0
    endDay = 363

    agentsDict, nameDict, networkDict, numAgents, loadingSeriesHP, chargingSeriesEV, \
    genSeriesPV, genSeriesWind, loadingSeriesDSM = util.agentsInit(alg, startDay, endDay, numAgentsEachType=1)
    nameList = [agent.id for agent in agentsDict.values()]
    typeList = [agent.type for agent in agentsDict.values()]
    grid = Grid(numAgents, nameList, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                         numCPU=20)

    sm = SpotMarket()
    agentsList = [obj for name, obj in agentsDict.items()]
    sm.addParticipants(agentsList)
    dso = DSO(grid, startDay, endDay)
    dso.addflexAgents(agentsList)

    """load the train Gym environment"""
    env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                         gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'alg': alg})
    env.gym.startDay = startDay
    env.gym.endDay = endDay

    env.reset()
    train_env = tf_py_environment.TFPyEnvironment(env)

    agents = train_env.pyenv.envs[0].gym.agents
    if not os.path.exists("../results/" + alg):
        os.makedirs("../results/" + alg)
    filename = "../results/" + alg + "/agentList.pkl"
    with open(filename, "wb") as f:
        pickle.dump(nameList, f)

    """Training parameters"""
    num_iterations = 1
    collect_steps_per_iteration = 365
    eval_interval = 5
    checkpoint_interval = 500
    batch_size = 1

    replay_buffer = util.replayBufferInit(train_env)
    time_step = train_env.reset()

    """to append and save for analysis"""
    totalReturns, loss = [], []

    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    time_step = train_env.reset()
    train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '../results/logs/' + alg + '/' + alg + '_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if alg == 'QMIX':
        nActions = train_env.action_spec()[0].shape[0]
        qmix = QMIX(nAgents=len(agents) * nActions, time_step_spec=train_env.time_step_spec(),
                    train_step_counter=train_step_counter,summary_writer=train_summary_writer)
    if alg == 'IQL':
        iql = IQL(networkDict=networkDict, nameDict=nameDict, nameList=nameList,
                  time_step_spec=train_env.time_step_spec(),
                  train_step_counter=train_step_counter,summary_writer=train_summary_writer)

    parameterDict = None
    """initialize the RL agents"""
    util.RLAgentInit(train_env, networkDict, nameDict, nameList, parameterDict, train_step_counter)
    """get the collection and evaluation policies"""
    collect_policySteps, eval_policySteps = util.getPolicies(networkDict, nameDict, nameList, alg)
    """initialize trainers for all network"""
    util.initializeTrainer(networkDict)

    """create checkpoint to resume training at a later stage"""
    checkpointDict = util.checkPointInit(nameList, nameDict, networkDict, replay_buffer, train_step_counter, alg)
    util.restoreCheckpoint(nameList, nameDict, networkDict, checkpointDict)
    print('Current training step:', train_step_counter)

    for num_iter in tqdm.trange(1, num_iterations + 1):
        """Collect a few steps using collect_policy and save to the replay buffer."""
        for _ in range(collect_steps_per_iteration):
            util.collect_step(train_env, collect_policySteps, replay_buffer, alg)
        """so that the reset step with 0 rewards is not added to the replay buffer"""
        if eval_interval>=(endDay-startDay):
            if num_iter%(endDay-startDay)==0:
                train_env.reset()

        """Sample a batch of data from the buffer and update the agent's network."""
        experience, unused_info = next(iterator)
        train_step_counter.assign_add(1)

        """training without multiprocessing"""
        if alg == 'MADDPG':
            train_loss = util.trainMADDPGAgents(experience, agents, nameDict, networkDict,
                                                nameList, typeList, train_summary_writer)
        elif alg == 'QMIX':
            train_loss = qmix.train(experience, agents, nameDict, networkDict).loss
        elif alg == 'IQL':
            train_loss = iql.train(experience, agents, nameDict, networkDict)
        print('Iteration: {} Loss: {}'.format(num_iter, train_loss))
        loss.append(train_loss.numpy())

        if num_iter % checkpoint_interval == 0:
            util.saveCheckpoint(nameDict, nameList, checkpointDict, train_step_counter, alg)

        if num_iter % eval_interval == 0:
            train_env.reset()
            total_return = util.compute_avg_return(train_env, eval_policySteps, alg, num_steps=endDay-startDay)
            totalReturns.append(total_return)
            """save the results for further evaluation"""
            util.saveToFile(totalReturns, loss, dso.congestionDetails, alg)
            totalReturns, loss = [], []
            train_env.reset()
    duration = (time.time()-st)/60
    print("---------------------------------------------%.2f minutes-----------------------------------" % duration)
    util.saveToFile([], [], dso.congestionDetails, alg)
