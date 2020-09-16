from EVehicle import EVehicle
from PVGen import PVG
from WindGen import WG
from HeatPump import HeatPump
from BatteryStorage import BatStorage
from DSM import DSM
from AgentNeuralNet import MADDPGAgent, QMIX
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
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import pickle
import tqdm
import optuna
import dask
import joblib
import dill


if __name__ == '__main__':
    st = time.time()

    # TODO why it is not reproducible even though np is seeded?
    np.random.seed(0)
    tf.random.set_seed(0)
    """specify the algorithm - MADDPG, QMIX"""
    alg = 'QMIX'

    agentsDict, nameDict, networkDict, numAgents, loadingSeriesHP, chargingSeriesEV, \
    genSeriesPV, genSeriesWind, loadingSeriesDSM = util.agentsInit(alg)
    grid = Grid(numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                         numCPU=20)

    sm = SpotMarket()
    agentsList = [obj for name, obj in agentsDict.items()]
    sm.addParticipants(agentsList)
    dso = DSO(grid)
    dso.addflexAgents(agentsList)

    # load the train Gym environment
    env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                         gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'alg': alg})
    env.reset()

    # convert to tf environment
    train_env = tf_py_environment.TFPyEnvironment(env)

    agents = train_env.pyenv.envs[0].gym.agents
    nameList = [agent.id for agent in agents]
    typeList = [agent.type for agent in agents]
    if alg == 'QMIX':
        qmix = QMIX(nAgents=len(agents), time_step_spec=train_env.time_step_spec())

    replay_buffer = util.replayBufferInit(train_env)
    batch_size = 8
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    """Training parameters"""
    num_iterations = 50
    collect_steps_per_iteration = 4
    eval_interval = 50
    time_step = train_env.reset()

    """to append and save for analysis"""
    congestionDFList = []
    lossList = []
    loss = []
    returns = []

    def objective(trial):
        parameterDict = util.hyperParameterOpt(trial)
        time_step = train_env.reset()
        train_step_counter = tf.Variable(0, trainable=False)

        # initialize the RL agents
        for node in networkDict:
            for type, network in networkDict[node].items():
                """name of the first agent in that particular type in this node"""
                name = nameDict[node][type][0]
                if len(network)>1:
                    for net in network:
                        net.hyperParameterInit(parameterDict)
                        net.initialize(train_env, train_step_counter, nameList.index(name))
                else:
                    network.hyperParameterInit(parameterDict)
                    network.initialize(train_env, train_step_counter, nameList.index(name))

        """This is the data collection policy"""
        collect_policySteps = []
        """This is the evaluation policy"""
        eval_policySteps = []
        for agentName in nameList:
            agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
            for type, names in nameDict[agentNode].items():
                if agentName in names:
                    if alg == 'MADDPG':
                        collect_policySteps.append(networkDict[agentNode][type].collect_policy.action)
                        eval_policySteps.append(networkDict[agentNode][type].eval_policy.action)
                        break
                    elif alg == 'QMIX':
                        collect_policySteps.extend([QAgent.collect_policy.action for QAgent in networkDict[agentNode][type]])
                        eval_policySteps.extend([QAgent.eval_policy.action for QAgent in networkDict[agentNode][type]])
                        break

        # initialize trainer
        networkList = []
        for node in networkDict:
            for net in networkDict[node].values():
                if len(net)>1:
                    networkList.extend(net)
                else:
                    networkList.append(net)

        for net in networkList:
            # (Optional) Optimize by wrapping some of the code in a graph using TF function.
            net.agent.train = common.function(net.agent.train)
            # Reset the train step
            net.agent.train_step_counter.assign(0)

        for num_iter in tqdm.trange(1, num_iterations + 1):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                util.collect_step(train_env, collect_policySteps, replay_buffer, alg)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            """total_agents_main and target are actions in case of MADDPG and Q values in case of QMIX"""
            time_steps, policy_steps, next_time_steps, total_agents_target, total_agents_main = \
                util.get_target_and_main_actions_or_values(experience, agents, nameDict, networkDict, alg)

            train_step_counter.assign_add(1)

            """training without multiprocessing"""
            if alg == 'MADDPG':
                for i, agentName in enumerate(nameList):
                    agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
                    train_loss = 0
                    for type, names in nameDict[agentNode].items():
                        if agentName in names:
                            train_loss = networkDict[agentNode][type].agent.train(time_steps, policy_steps, next_time_steps,
                                                                                  total_agents_target,
                                                                                  total_agents_main,
                                                                                  index=i).loss
                            break
                    step = networkDict[agentNode][typeList[i]].agent.train_step_counter.numpy()
            elif alg == 'QMIX':
                train_loss = qmix.train(time_steps, policy_steps, next_time_steps, total_agents_target,
                                                                                  total_agents_main).loss

            """unpromising trials at the early stages of the training"""
            intermediate_return = util.compute_avg_return(train_env, eval_policySteps, alg, num_steps=2)
            trial.report(intermediate_return.sum(), num_iter)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if num_iter % eval_interval == 0:
                avg_return = util.compute_avg_return(train_env, eval_policySteps, alg, num_steps=2)
                print('step: {0}, Avg Return: {1}'.format(num_iter, avg_return))
                print("=====================================================================================")
                joblib.dump(study, '../results/study.pkl')
                return avg_return.sum()

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                               n_warmup_steps=20))
    study.optimize(objective, n_trials=10)

    duration = (time.time()-st)/60
    print("---------------------------------------------%.2f minutes-----------------------------------" % duration)
