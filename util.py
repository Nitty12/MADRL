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


def agentsInit():
    path = os.getcwd()
    if os.path.isfile("../inputs/RA_RD_Import_.pkl"):
        data = pd.read_pickle("../inputs/RA_RD_Import_.pkl")
    else:
        datapath = os.path.join(path, "../inputs/RA_RD_Import_.csv")
        data = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0,
                           usecols=['Name', 'Location', 'Un'], error_bad_lines=False)
        data.columns = ['Name', 'Location', 'Un_kV']
        data.reset_index(inplace=True, drop=True)
        data['Un_kV'] = data['Un_kV'].apply(pd.to_numeric)
        data.to_pickle("../inputs/RA_RD_Import_.pkl")

    loadingSeriesHP = getHPSeries()
    chargingSeriesEV, capacitySeriesEV, absenceSeriesEV, consumptionSeriesEV = getEVSeries()
    relativePathCSV = "../inputs/PV_Zeitreihe_nnf_1h.csv"
    relativePathPickle = "../inputs/PV_Zeitreihe_nnf_1h.pkl"
    genSeriesPV = getGenSeries(relativePathCSV, relativePathPickle)
    relativePathCSV = "../inputs/WEA_nnf_1h.csv"
    relativePathPickle = "../inputs/WEA_nnf_1h.pkl"
    genSeriesWind = getGenSeries(relativePathCSV, relativePathPickle)
    loadingSeriesDSM = getDSMSeries()

    """contains names of the respective agents"""
    homeStorageList = []
    for name in data['Name']:
        if name.endswith('_nsHeimSpeicherErsatzeinsp'):
            homeStorageList.append(name)

    agentsDict = {}
    """the number of agents in each type to consider as flexibility"""
    numAgents = 1
    """negate the PV and Wind timeseries to make generation qty -ve"""
    for name in genSeriesPV.columns[:numAgents]:
        name = re.search('k.*', name).group(0)
        loc, voltage_level, min_power, max_power = getAgentDetails(data, name)
        colName = genSeriesPV.filter(like=name).columns.values[0]
        agentsDict[name] = PVG(id=name, location=loc, minPower=min_power, maxPower=max_power,
                               voltageLevel=voltage_level, genSeries=genSeriesPV.loc[:, colName])
    for name in genSeriesWind.columns[:numAgents]:
        name = re.search('k.*', name).group(0)
        loc, voltage_level, min_power, max_power = getAgentDetails(data, name)
        colName = genSeriesWind.filter(like=name).columns.values[0]
        agentsDict[name] = WG(id=name, location=loc, minPower=min_power, maxPower=max_power,
                               voltageLevel=voltage_level, genSeries=genSeriesWind.loc[:, colName])
    for name in homeStorageList[:numAgents]:
        name = re.search('k.*', name).group(0)
        loc, voltage_level, min_power, max_power = getAgentDetails(data, name)
        agentsDict[name] = BatStorage(id=name, location=loc, minPower=min_power, maxPower=max_power,
                                      voltageLevel=voltage_level, maxCapacity=10*max_power, marginalCost = 30)
    for name in chargingSeriesEV.columns[:numAgents]:
        name = re.search('k.*', name).group(0)
        colName = capacitySeriesEV.filter(like=name[:-5]).columns.values[0]
        agentsDict[name] = EVehicle(id=name, maxCapacity=capacitySeriesEV.loc[0, colName],
                                    absenceTimes=absenceSeriesEV.loc[:, colName],
                                    consumption=consumptionSeriesEV.loc[:, colName], marginalCost = 30)
    for name in loadingSeriesHP.columns[:numAgents]:
        name = re.search('k.*', name).group(0)
        #TODO check if the latest RA_RD_Import_ file contains maxpower
        colName = loadingSeriesHP.filter(like=name).columns.values[0]
        agentsDict[name] = HeatPump(id=name, maxPower=round(loadingSeriesHP.loc[:, colName].max(),5),
                                    maxStorageLevel=25*round(loadingSeriesHP.loc[:, colName].max(),5),
                                    scheduledLoad=loadingSeriesHP.loc[:, colName], marginalCost = 30)
    for name in loadingSeriesDSM.columns[:numAgents]:
        name = re.search('k.*', name).group(0)
        #TODO check if the latest RA_RD_Import_ file contains maxpower
        colName = loadingSeriesDSM.filter(like=name).columns.values[0]
        agentsDict[name] = DSM(id=name, maxPower=round(loadingSeriesDSM.loc[:, colName].max(),5),
                               scheduledLoad=loadingSeriesDSM.loc[:, colName], marginalCost = 30)

    nameDict, networkDict = RLNetworkInit(agentsDict)

    return agentsDict, nameDict, networkDict, numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, \
           loadingSeriesDSM


def RLNetworkInit(agentsDict):
    """parameter sharing of RL Network for same types of agents in same node"""
    networkDict = {}
    nameDict = {}
    for name, agent in agentsDict.items():
        agent.node = 'Standort_' + re.search("k(\d+)[n,d,l]", agent.id).group(1)
        if networkDict.get(agent.node) is None:
            networkDict[agent.node] = {}
            nameDict[agent.node] = {}
        if nameDict[agent.node].get(agent.type) is None:
            nameDict[agent.node][agent.type] = [name]
        else:
            nameDict[agent.node][agent.type].append(name)
        if networkDict[agent.node].get(agent.type) is None:
            networkDict[agent.node][agent.type] = AgentNeuralNet()
    return nameDict, networkDict


def getEVSeries():
    path = os.getcwd()
    """EV timeseries"""
    if os.path.isfile("../inputs/EMob_Zeitreihe_nnf_1h.pkl"):
        chargingSeries = pd.read_pickle("../inputs/EMob_Zeitreihe_nnf_1h.pkl")
    else:
        datapath = os.path.join(path, "../inputs/EMob_Zeitreihe_nnf_1h.csv")
        chargingSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape')
        chargingSeries.drop('Unnamed: 0', axis=1, inplace=True)
        chargingSeries = chargingSeries.apply(pd.to_numeric)
        chargingSeries.to_pickle("../inputs/EMob_Zeitreihe_nnf_1h.pkl")

    """EV capacity timeseries"""
    if os.path.isfile("../inputs/EMob_Zeitreihe_capacity.pkl"):
        capacitySeries = pd.read_pickle("../inputs/EMob_Zeitreihe_capacity.pkl")
    else:
        datapath = os.path.join(path, "../inputs/EMob_Zeitreihe_capacity.csv")
        capacitySeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape')
        capacitySeries.drop('Unnamed: 0', axis=1, inplace=True)
        capacitySeries = capacitySeries.apply(pd.to_numeric)
        capacitySeries.to_pickle("../inputs/EMob_Zeitreihe_capacity.pkl")

    """EV absence timeseries"""
    if os.path.isfile("../inputs/EMob_Zeitreihe_absence.pkl"):
        absenceSeries = pd.read_pickle("../inputs/EMob_Zeitreihe_absence.pkl")
    else:
        datapath = os.path.join(path, "../inputs/EMob_Zeitreihe_absence.csv")
        absenceSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape')
        absenceSeries.drop('Unnamed: 0', axis=1, inplace=True)
        absenceSeries.columns = capacitySeries.columns
        absenceSeries.to_pickle("../inputs/EMob_Zeitreihe_absence.pkl")

    """EV consumption timeseries"""
    if os.path.isfile("../inputs/EMob_Zeitreihe_consumption.pkl"):
        consumptionSeries = pd.read_pickle("../inputs/EMob_Zeitreihe_consumption.pkl")
    else:
        datapath = os.path.join(path, "../inputs/EMob_Zeitreihe_consumption.csv")
        consumptionSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                        encoding='unicode_escape')
        consumptionSeries.drop('Unnamed: 0', axis=1, inplace=True)
        consumptionSeries.to_pickle("../inputs/EMob_Zeitreihe_consumption.pkl")
    return chargingSeries, capacitySeries, absenceSeries, consumptionSeries


def getHPSeries():
    """Heat pump loading timeseries"""
    path = os.getcwd()
    if os.path.isfile("../inputs/WP_Zeitreihe_nnf.pkl"):
        loadingSeries = pd.read_pickle("../inputs/WP_Zeitreihe_nnf.pkl")
    else:
        datapath = os.path.join(path, "../inputs/WP_Zeitreihe_nnf.csv")
        loadingSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape')
        """cleaning the dataframe"""
        loadingSeries.drop('NNF', axis=1, inplace=True)
        loadingSeries.drop(loadingSeries.index[0], inplace=True)
        # TODO check if there is this extra row in every HP series
        loadingSeries.drop(loadingSeries.index[8760], inplace=True)
        loadingSeries.reset_index(drop=True, inplace=True)
        loadingSeries = editNames(loadingSeries)
        loadingSeries = loadingSeries.apply(pd.to_numeric)
        loadingSeries.to_pickle("../inputs/WP_Zeitreihe_nnf.pkl")
    return loadingSeries


def getDSMSeries():
    """DSM scheduled load timeseries"""
    path = os.getcwd()
    if os.path.isfile("../inputs/ang_Kunden_GHD_nnf_1h.pkl"):
        loadingSeries = pd.read_pickle("../inputs/ang_Kunden_GHD_nnf_1h.pkl")
    else:
        datapath = os.path.join(path, "../inputs/ang_Kunden_GHD_nnf_1h.csv")
        loadingSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape')
        """cleaning the dataframe"""
        loadingSeries.drop('NNF', axis=1, inplace=True)
        loadingSeries.drop(loadingSeries.index[0], inplace=True)
        loadingSeries.reset_index(drop=True, inplace=True)
        loadingSeries = editNames(loadingSeries)
        loadingSeries = loadingSeries.apply(pd.to_numeric)
        loadingSeries.to_pickle("../inputs/ang_Kunden_GHD_nnf_1h.pkl")
    return loadingSeries


def editNames(loadingSeries):
    if not loadingSeries.columns[0].endswith('_flex'):
        colNames = []
        for name in loadingSeries:
            colNames.append(name+'_flex')
        loadingSeries.columns = colNames
    return loadingSeries


def getGenSeries(relativePathCSV, relativePathPickle):
    """get PV and wind generation timeseries"""
    path = os.getcwd()
    if os.path.isfile(relativePathPickle):
        genSeries = pd.read_pickle(relativePathPickle)
    else:
        datapath = os.path.join(path, relativePathCSV)
        genSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                encoding='unicode_escape', nrows=0)
        genSeries.drop('NNF', axis=1, inplace=True)
        columnNames = list(genSeries)
        genSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                encoding='unicode_escape', dtype=float)
        genSeries.drop('NNF', axis=1, inplace=True)
        genSeries.columns = columnNames
        """converting to negative value for generation"""
        genSeries = -genSeries
        genSeries.to_pickle("../inputs/ang_Kunden_GHD_nnf_1h.pkl")
    return genSeries


def getAgentDetails(data, name):
    details = data.loc[data['Name'] == name]
    loc = details['Location'].values[0]
    voltage_level = details['Un_kV'].values[0]
    """no information on maximum and minimum power in the latest csv"""
    min_power = 2
    max_power = 2
    return loc, voltage_level, min_power, max_power


def replayBufferInit(train_env):
    """replay buffer stores transition of all the agents"""
    replay_buffer_capacity = 100000
    policy_step_spec = policy_step.PolicyStep(action=train_env.action_spec(),
                                              state=(),
                                              info=())
    replay_buffer_data_spec = trajectory.from_transition(train_env.time_step_spec(),
                                                         policy_step_spec,
                                                         train_env.time_step_spec())
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=replay_buffer_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=replay_buffer_capacity)
    return replay_buffer


def one_step(environment, policySteps):
    """iteration alternate between spot and flex states"""
    total_agents_action = []
    time_step = environment.current_time_step()
    for i, policyStep in enumerate(policySteps):
        individual_time_step = ts.get_individual_time_step(time_step, i)
        action = policyStep(individual_time_step)
        total_agents_action.append(action)
    next_time_step = environment.step(tuple(total_agents_action))
    return time_step, total_agents_action, next_time_step


def compute_avg_return(environment, policySteps, num_steps=10):
    total_return = None
    for step in range(num_steps):
        _, _, next_time_step = one_step(environment, policySteps)
        if step == 0:
            total_return = next_time_step.reward
        else:
            total_return += next_time_step.reward
    avg_return = total_return / num_steps
    return avg_return.numpy()[0]


def collect_step(environment, policySteps, buffer):
    time_step, total_agents_action, next_time_step = one_step(environment, policySteps)
    traj = trajectory.from_transition(time_step, total_agents_action, next_time_step, joint_action=True)
    buffer.add_batch(traj)
    # tf_metrics.AverageReturnMetric(traj)


def get_target_and_main_actions(experience, agents, nameDict, networkDict):
    """get the actions from the target actor network and main actor network of all the agents"""
    total_agents_target_actions = []
    total_agents_main_actions = []
    time_steps, policy_steps, next_time_steps = (
        trajectory.experience_to_transitions(experience, squeeze_time_dim=True))
    for i, flexAgent in enumerate(agents):
        for node in nameDict:
            target_action = None
            for type, names in nameDict[node].items():
                if flexAgent.id in names:
                    target_action, _ = networkDict[node][type].agent._target_actor_network(next_time_steps.observation[i],
                                                                    next_time_steps.step_type,
                                                                    training=False)
                    main_action, _ = networkDict[node][type].agent._actor_network(time_steps.observation[i],
                                                                       time_steps.step_type,
                                                                       training=True)
                    break
            if target_action is not None:
                break
        total_agents_target_actions.append(target_action)
        total_agents_main_actions.append(main_action)
    return time_steps, policy_steps, next_time_steps, \
           tuple(total_agents_target_actions), tuple(total_agents_main_actions)


def checkPointInit(nameList, nameDict, networkDict, replay_buffer, train_step_counter):
    """to save and restore the trained RL agents"""
    path = os.getcwd()
    checkpointDict = {}
    for i, agentName in enumerate(nameList):
        agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
        for type, names in nameDict[agentNode].items():
            if agentName in names:
                relativePath = '../results/checkpointAgent_' + agentName
                checkpointDict[agentName] = common.Checkpointer(
                    ckpt_dir=os.path.join(path, relativePath),
                    max_to_keep=1,
                    agent=networkDict[agentNode][type].agent,
                    policy=networkDict[agentNode][type].agent.policy,
                    global_step=train_step_counter
                )
                break
    relativePath = '../results/checkpointReplayBuffer'
    checkpointDict['ReplayBuffer'] = common.Checkpointer(
        ckpt_dir=os.path.join(path, relativePath),
        max_to_keep=1,
        replay_buffer=replay_buffer
    )
    return checkpointDict


def restoreCheckpoint(nameList, nameDict, checkpointDict):
    checkpointDict['ReplayBuffer'].initialize_or_restore()
    for i, agentName in enumerate(nameList):
        agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
        for type, names in nameDict[agentNode].items():
            if agentName in names:
                checkpointDict[agentName].initialize_or_restore()
                break
    train_step_counter = tf.compat.v1.train.get_global_step()


def trainLoop(agentIndexAndName, nameDict, networkDict, time_steps, policy_steps, next_time_steps, total_agents_target_actions,
              total_agents_main_actions, num_iter, checkpointInterval, checkpointDict, train_step_counter, typeList,
              log_interval, loss):
    agentName = agentIndexAndName[1]
    index = agentIndexAndName[0]
    agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentName).group(1)
    train_loss = 0
    for type, names in nameDict[agentNode].items():
        if agentName in names:
            train_loss = networkDict[agentNode][type].agent.train(time_steps, policy_steps, next_time_steps,
                                                                  total_agents_target_actions,
                                                                  total_agents_main_actions,
                                                                  index=index).loss
            if num_iter % checkpointInterval == 0:
                """save the training state of all agents"""
                checkpointDict[agentName].save(train_step_counter)
            break
    step = networkDict[agentNode][typeList[index]].agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('ID: {0}, step: {1}, loss: {2}'.format(agentName, step, train_loss))
        loss.append((index, train_loss.numpy()))
    return loss


def hyperParameterOpt(trial):
    """hyperparameter optimization with optuna"""
    parameterDict = {}
    parameterDict['learning_rate'] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1)
    parameterDict['discount_factor'] = trial.suggest_float('discount_factor', 0.95, 0.999)

    parameterDict['actor_activation_fn'] = trial.suggest_categorical('actor_activation_fn',
                                                         ['relu', 'leaky_relu'])
    parameterDict['critic_activation_fn'] = trial.suggest_categorical('critic_activation_fn',
                                                         ['relu', 'leaky_relu'])

    parameterDict['fc_layer_params_actor'] = []
    parameterDict['fc_dropout_layer_params_actor'] = []
    actor_n_layers = trial.suggest_int('actor_n_layers', 1, 3)
    for i in range(actor_n_layers):
        num_hidden = trial.suggest_int("actor_n_units_L{}".format(i+1), 50, 300)
        dropout_rate = trial.suggest_float("actor_dropout_rate_L{}".format(i+1), 0, 0.5)
        parameterDict['fc_layer_params_actor'].append(num_hidden)
        parameterDict['fc_dropout_layer_params_actor'].append(dropout_rate)

    parameterDict['fc_layer_params_critic_obs'] = []
    parameterDict['fc_dropout_layer_params_critic_obs'] = []
    critic_obs_n_layers = trial.suggest_int('critic_obs_n_layers', 1, 2)
    for i in range(critic_obs_n_layers):
        num_hidden = trial.suggest_int("critic_obs_n_units_L{}".format(i+1), 50, 300)
        dropout_rate = trial.suggest_float("critic_obs_dropout_rate_L{}".format(i+1), 0, 0.5)
        parameterDict['fc_layer_params_critic_obs'].append(num_hidden)
        parameterDict['fc_dropout_layer_params_critic_obs'].append(dropout_rate)

    parameterDict['fc_layer_params_critic_merged'] = []
    parameterDict['fc_dropout_layer_params_critic_merged'] = []
    critic_merged_n_layers = trial.suggest_int('critic_merged_n_layers', 1, 2)
    for i in range(critic_merged_n_layers):
        num_hidden = trial.suggest_int("critic_merged_n_units_L{}".format(i+1), 50, 300)
        dropout_rate = trial.suggest_float("critic_merged_dropout_rate_L{}".format(i+1), 0, 0.5)
        parameterDict['fc_layer_params_critic_merged'].append(num_hidden)
        parameterDict['fc_dropout_layer_params_critic_merged'].append(dropout_rate)

    return parameterDict