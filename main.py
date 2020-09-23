from AgentNeuralNet import MADDPGAgent, QMIX, IQL
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
import datetime
pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    st = time.time()
    np.random.seed(0)
    tf.random.set_seed(0)

    """specify the algorithm - MADDPG, QMIX, IQL"""
    alg = 'MADDPG'

    agentsDict, nameDict, networkDict, numAgents, loadingSeriesHP, chargingSeriesEV, \
    genSeriesPV, genSeriesWind, loadingSeriesDSM = util.agentsInit(alg)
    grid = Grid(numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                         numCPU=20)

    sm = SpotMarket()
    agentsList = [obj for name, obj in agentsDict.items()]
    sm.addParticipants(agentsList)
    dso = DSO(grid)
    dso.addflexAgents(agentsList)

    """load the train Gym environment"""
    env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                         gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'alg': alg})
    env.reset()
    train_env = tf_py_environment.TFPyEnvironment(env)

    agents = train_env.pyenv.envs[0].gym.agents
    nameList = [agent.id for agent in agents]
    typeList = [agent.type for agent in agents]

    """Training parameters"""
    num_iterations = 50
    collect_steps_per_iteration = 4
    eval_interval = 50
    batch_size = 8

    replay_buffer = util.replayBufferInit(train_env)
    time_step = train_env.reset()

    """to append and save for analysis"""
    congestionDFList = []
    lossList = []
    loss = []
    returns = []

    def objective(trial):
        parameterDict = util.hyperParameterOpt(trial, alg)
        batch_size = parameterDict['batch_size']
        dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
        iterator = iter(dataset)

        time_step = train_env.reset()
        train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = '../results/logs/' + alg + '_' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if alg == 'QMIX':
            nActions = train_env.action_spec()[0].shape[0]
            qmix = QMIX(nAgents=len(agents) * nActions, time_step_spec=train_env.time_step_spec(),
                        train_step_counter=train_step_counter,summary_writer=train_summary_writer)
            qmix.hyperParameterInit(parameterDict)
        if alg == 'IQL':
            iql = IQL(networkDict=networkDict, nameDict=nameDict, nameList=nameList,
                      time_step_spec=train_env.time_step_spec(),
                      train_step_counter=train_step_counter,summary_writer=train_summary_writer)
        """initialize the RL agents"""
        util.RLAgentInit(train_env, networkDict, nameDict, nameList, parameterDict, train_step_counter)
        """get the collection and evaluation policies"""
        collect_policySteps, eval_policySteps = util.getPolicies(networkDict, nameDict, nameList, alg)
        """initialize trainers for all network"""
        util.initializeTrainer(networkDict)
        # """create checkpoint to resume training at a later stage"""
        # checkpointDict = util.checkPointInit(nameList, nameDict, networkDict, replay_buffer, train_step_counter, alg)
        # util.restoreCheckpoint(nameList, nameDict, networkDict, checkpointDict)

        for num_iter in tqdm.trange(1, num_iterations + 1):
            """Collect a few steps using collect_policy and save to the replay buffer."""
            for _ in range(collect_steps_per_iteration):
                util.collect_step(train_env, collect_policySteps, replay_buffer, alg)

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


            if num_iter % eval_interval == 0:
                avg_return = util.compute_avg_return(train_env, eval_policySteps, alg, num_steps=4)
                print('step: {0}, Avg Return: {1}'.format(num_iter, avg_return))
                print("=====================================================================================")
                study_path = '../results/study'+alg+'.pkl'
                joblib.dump(study, study_path)
                return avg_return.sum()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    duration = (time.time()-st)/60
    print("---------------------------------------------%.2f minutes-----------------------------------" % duration)
