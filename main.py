from EVehicle import EVehicle
from PVGen import PVG
from WindGen import WG
from HeatPump import HeatPump
from BatteryStorage import BatStorage
from DSM import DSM
from AgentNeuralNet import AgentNeuralNet
from SpotMarket import SpotMarket
from DSO import DSO
from Grid import Grid
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments import suite_gym
import os
import time
import re
import util
from multiprocessing import Pool
import pickle
import tqdm

if __name__ == '__main__':
    st = time.time()

    # TODO why it is not reproducible even though np is seeded?
    np.random.seed(0)
    tf.random.set_seed(0)
    '''
    generation is -ve qty and load is +ve qty
    '''

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    agentsDict, nameDict, networkDict, numAgents, loadingSeriesHP, chargingSeriesEV, \
    genSeriesPV, genSeriesWind, loadingSeriesDSM = util.agentsInit()
    grid = Grid(numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                         numCPU=24)

    sm = SpotMarket()
    agentsList = [obj for name, obj in agentsDict.items()]
    sm.addParticipants(agentsList)

    dso = DSO(grid)
    dso.addflexAgents(agentsList)

    # load the train Gym environment
    env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                         gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'grid': grid})
    env.reset()

    # convert to tf environment
    train_env = tf_py_environment.TFPyEnvironment(env)

    time_step = train_env.reset()

    replay_buffer = util.replayBufferInit(train_env)
    train_step_counter = tf.Variable(0, trainable=False)
    batch_size = 8
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # initialize the ddpg agents
    agents = train_env.pyenv.envs[0].gym.agents
    nameList = [agent.id for agent in agents]
    typeList = [agent.type for agent in agents]
    for node in networkDict:
        for type, network in networkDict[node].items():
            """name of the first agent in that particular type in this node"""
            name = nameDict[node][type][0]
            network.initialize(train_env, train_step_counter, nameList.index(name))

    """This is the data collection policy"""
    collect_policySteps = []
    """This is the evaluation policy"""
    eval_policySteps = []
    for agentName in nameList:
        agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
        for type, names in nameDict[agentNode].items():
            if agentName in names:
                collect_policySteps.append(networkDict[agentNode][type].collect_policy.action)
                eval_policySteps.append(networkDict[agentNode][type].eval_policy.action)
                break

    # initialize trainer
    networkList = [net for node in networkDict for net in networkDict[node].values()]
    for net in networkList:
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        net.agent.train = common.function(net.agent.train)
        # Reset the train step
        net.agent.train_step_counter.assign(0)

    # Training the agents
    num_iterations = 10
    collect_steps_per_iteration = 2
    log_interval = 2
    eval_interval = 5
    checkpointInterval = 10
    time_step = train_env.reset()

    """to append and save for analysis"""
    congestionDFList = []
    lossList = []
    loss = []

    """create checkpoint to resume training at a later stage"""
    checkpointDict = util.checkPointInit(nameList, nameDict, networkDict, replay_buffer, train_step_counter)
    util.restoreCheckpoint(nameList, nameDict, checkpointDict)

    # Evaluate the agent's policy once before training.
    avg_return = util.compute_avg_return(train_env, eval_policySteps, num_steps=10)
    returns = [avg_return]

    for num_iter in tqdm.trange(1, num_iterations + 1):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            util.collect_step(train_env, collect_policySteps, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        time_steps, policy_steps, next_time_steps, total_agents_target_actions, total_agents_main_actions = \
            util.get_target_and_main_actions(experience, agents, nameDict, networkDict)

        train_step_counter.assign_add(1)

        for i, agentName in enumerate(nameList):
            agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
            train_loss = 0
            for type, names in nameDict[agentNode].items():
                if agentName in names:
                    train_loss = networkDict[agentNode][type].agent.train(time_steps, policy_steps, next_time_steps,
                                                                          total_agents_target_actions,
                                                                          total_agents_main_actions,
                                                                          index=i).loss
                    if num_iter % checkpointInterval == 0:
                        """save the training state of all agents"""
                        checkpointDict[agentName].save(train_step_counter)
                    break
            step = networkDict[agentNode][typeList[i]].agent.train_step_counter.numpy()
            if step % log_interval == 0:
                print('ID: {0}, step: {1}, loss: {2}'.format(agentName, step, train_loss))
                loss.append(train_loss.numpy())
        lossList.append(loss)
        loss = []
        print('iteration: ', num_iter)

        if num_iter % checkpointInterval == 0:
            """save the Replay Buffer"""
            checkpointDict['ReplayBuffer'].save(train_step_counter)

        if train_step_counter % log_interval == 0:
            print("=====================================================================================")

        if num_iter % eval_interval == 0:
            avg_return = util.compute_avg_return(train_env, eval_policySteps, num_steps=10)
            print('step: {0}, Avg Return: {1}'.format(num_iter, avg_return))
            print("=====================================================================================")
            returns.append(avg_return)

    """Plot the returns"""
    steps = range(0, num_iterations + 1, eval_interval)
    idx = 0
    agent_returns = [ret[idx] for ret in returns]
    plt.plot(steps, agent_returns)
    plt.ylabel('Avg Return - Agent {}'.format(idx))
    plt.xlabel('Step')
    plt.show()

    """save the results for further evaluation"""
    with open("../results/returns.pkl", "ab") as f:
        pickle.dump(returns, f)
    with open("../results/loss.pkl", "ab") as f:
        pickle.dump(lossList, f)
    with open("../results/agentList.pkl", "wb") as f:
        pickle.dump(nameList, f)

    duration = (time.time()-st)/60
    print("---------------------------------------------%.2f minutes-----------------------------------" % duration)
