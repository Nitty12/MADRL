import numpy as np
import pandas as pd
import os
import re
from multiprocessing import Pool
import pickle
import time as t
import util


class Grid:
    def __init__(self,numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                 TimePeriod=8760, numNodes=0, numLines=0, numCPU=24):
        """the number of agents in each type to consider as flexibility"""
        self.numAgents = numAgents
        self.loadingSeriesHP = loadingSeriesHP
        self.chargingSeriesEV = chargingSeriesEV
        self.genSeriesPV = genSeriesPV
        self.genSeriesWind = genSeriesWind
        self.loadingSeriesDSM = loadingSeriesDSM
        self.TimePeriod = TimePeriod
        self.numNodes = numNodes
        self.numLines = numLines
        self.nodes = []
        self.lines = []
        self.agentNodes = []
        self.data = None
        self.loading = None
        self.HVTrafoNode = None

        """for loading sensis"""
        self.agentsWithNewNodeList = []
        self.nodeSensitivityList = []
        self.numCPU = numCPU

        self.reqdFlexTimes = np.array([])
        self.flexAgents = []
        self.congestionStatus = None
        """contains the total load per node for the households"""
        self.loadsAndGens = None
        """contains the list of households in each node"""
        self.nodeDict = None
        """contains the sensitivity matrix with index as trafo/lines and columns as flexagents"""
        self.sensitivity = None
        """contains the flexagent name connected to a particular node to get the sensitivity to use for 
        power flow approximation"""
        self.nodeSensitivityDict = {}
        self.day = 0
        self.dailyFlexTime = 24
        self.dailyTimes = None
        self.importGrid()
        self.reset()

    def reset(self):
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)

    def importGrid(self):
        """get the grid data from the CSV"""
        path = os.getcwd()
        if os.path.isfile("../inputs/CBCO_Results_.pkl"):
            self.data = pd.read_pickle("../inputs/CBCO_Results_.pkl")
        else:
            datapath = os.path.join(path, "../inputs/CBCO_Results_.csv")
            self.data = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=1, error_bad_lines=False)
            """skip the unwanted rows in between"""
            self.data = self.data.loc[4:,
                   ['Name', 'Location From', 'Un From', 'Location To', 'Un To', 'P N-0 before Opt.', 'I N-0 before Opt.',
                    'Loading N-0 before Opt.']]
            self.data.columns = ['Name', 'Loc_from', 'Un_from_kV', 'Loc_to', 'Un_to_kV', 'P_MW', 'I_A', 'Loading_percent']
            self.data.reset_index(inplace=True, drop=True)
            self.data[['Un_from_kV', 'Un_to_kV', 'P_MW', 'I_A', 'Loading_percent']] = self.data[
                ['Un_from_kV', 'Un_to_kV', 'P_MW', 'I_A', 'Loading_percent']].apply(pd.to_numeric)
            """Calculate the rated current from the data"""
            ratedI = self.data.I_A / (0.01 * self.data.Loading_percent)
            self.data = self.data.assign(I_rated_A=ratedI)
            """seems like all current measurements are in the LV side,
                so multiplying with the turns ratio 110/10 and 10/.4"""
            self.data.loc[self.data['Name']=='Trafo_HSMS', 'I_rated_A'] *= 11
            self.data.loc[self.data['Name']=='Trafo_HSMS_par', 'I_rated_A'] *= 11
            LVTrafos = self.data.loc[self.data['Un_to_kV']==0.4, 'Name'].values
            self.data.loc[self.data['Name'].isin(LVTrafos), 'I_rated_A'] *= 25
            self.data.to_pickle("../inputs/CBCO_Results_.pkl")

        for name in self.data['Name']:
            if name.startswith(('Trafo', 'Knoten')):
                self.nodes.append(name)
            elif name.startswith('L'):
                self.lines.append(name)
        self.numNodes = len(self.nodes)
        self.numLines = len(self.lines)

    def importLoadsAndSensi(self):
        """get the household load data from the CSV"""
        self.getLoadsAndGens()
        """load the sensitivity matrix and concatenate"""
        self.loadSensitivityMatrix()
        """retain only the lines and nodes present in the sensitivity matrix"""
        linesAndNodes = self.sensitivity.index[self.sensitivity['time_step'] == 1].values
        self.data = self.data.loc[self.data['Name'].isin(linesAndNodes),:]
        self.loading = pd.DataFrame(np.full((self.dailyFlexTime, len(linesAndNodes)), 0.0),
                                    columns=linesAndNodes, index=self.dailyTimes)
        self.congestionStatus = pd.DataFrame(np.full((self.dailyFlexTime, len(linesAndNodes)), False),
                                             columns=linesAndNodes, index=self.dailyTimes)

    def getLoadsAndGens(self):
        path = os.getcwd()
        if os.path.isfile("../inputs/loadingSeriesHH.pkl"):
            loadingSeriesHH = pd.read_pickle("../inputs/loadingSeriesHH.pkl")
            columnNamesHH = list(loadingSeriesHH.columns)
        else:
            """reading household loads"""
            datapath = os.path.join(path, "../inputs/ang_Kunden_HH1_nnf_corrected_1h.csv")
            loadingSeriesColumns = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                        encoding='unicode_escape', nrows=0)
            loadingSeriesColumns.drop('NNF', axis=1, inplace=True)
            columnNamesHH1 = list(loadingSeriesColumns)
            loadingSeriesHH = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                        encoding='unicode_escape', dtype=float)
            loadingSeriesHH.drop('NNF', axis=1, inplace=True)
            loadingSeriesHH.columns = columnNamesHH1

            datapath = os.path.join(path, "../inputs/ang_Kunden_HH2_nnf_corrected_1h.csv")
            loadingSeriesColumns = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                         encoding='unicode_escape', nrows=0)
            loadingSeriesColumns.drop('NNF', axis=1, inplace=True)
            columnNamesHH2 = list(loadingSeriesColumns)
            loadingSeriesHH2 = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                         encoding='unicode_escape', dtype=float)
            loadingSeriesHH2.drop('NNF', axis=1, inplace=True)
            loadingSeriesHH2.columns = columnNamesHH2
            loadingSeriesHH = pd.concat([loadingSeriesHH, loadingSeriesHH2], axis=1, sort=False)
            columnNamesHH = columnNamesHH1 + columnNamesHH2
            # TODO delete efficiently may be use multiprocess pool etc
            del loadingSeriesHH2
            loadingSeriesHH.to_pickle("../inputs/loadingSeriesHH.pkl")

        columnNames = columnNamesHH + \
                      list(self.loadingSeriesHP.columns[self.numAgents:]) + \
                      list(self.chargingSeriesEV.columns[self.numAgents:]) + \
                      list(self.loadingSeriesDSM.columns[self.numAgents:]) + \
                      list(self.genSeriesPV.columns[self.numAgents:]) + \
                      list(self.genSeriesWind.columns[self.numAgents:])

        """get the list of nodes in which the loads and generations are connected"""
        nodes = []
        for name in columnNames:
            n = re.search("(Standort_[0-9]+)", name)
            if not n.group() in nodes:
                nodes.append(n.group())

        """get the list of loads and gens in each node
            and combine these for the same node and only store the result"""
        self.loadsAndGens = pd.DataFrame(columns=nodes)
        nodeDict = {}
        for node in nodes:
            nodeDict[node] = []
            for name in columnNames:
                match = re.search(rf".*{node}.*", name)
                if match:
                    nodeDict[node].append(match.group())
            self.loadsAndGens[node] = loadingSeriesHH.loc[:, loadingSeriesHH.columns.isin(nodeDict[node])].sum(axis=1) + \
                                      self.loadingSeriesHP.loc[:, self.loadingSeriesHP.columns.isin(nodeDict[node])].sum(axis=1) + \
                                      self.chargingSeriesEV.loc[:, self.chargingSeriesEV.columns.isin(nodeDict[node])].sum(axis=1) + \
                                      self.loadingSeriesDSM.loc[:, self.loadingSeriesDSM.columns.isin(nodeDict[node])].sum(axis=1) + \
                                      self.genSeriesPV.loc[:, self.genSeriesPV.columns.isin(nodeDict[node])].sum(axis=1) + \
                                      self.genSeriesWind.loc[:, self.genSeriesWind.columns.isin(nodeDict[node])].sum(axis=1)

    def loadSensitivityMatrix(self):
        """walk through all the sensitivity matrices and concatenate as a single dataframe"""
        path = os.getcwd()
        datapath = os.path.join(path, "../inputs/sensitivity")
        fileList = [os.path.join(root, file) for root, dirs, files in os.walk(datapath) for file in files if
                    file.endswith('Sensitivities_Init_.csv')]

        if os.path.isfile("../inputs/SensiDFNew.pkl"):
            print("Reading sensitivity dataframe...")
            self.sensitivity = pd.read_pickle("../inputs/SensiDFNew.pkl")

            """Read the first file to get required data useful for consequent processing"""
            df = pd.read_csv(fileList[0], sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                             encoding='unicode_escape')
            dfList = []
            if not dfList:
                """gets the first flexagent name connected to a particular node to get the sensitivity to use for
                    power flow approximation"""
                for node in self.loadsAndGens:
                    nodeNumber = node[9:]
                    for colName in df:
                        match = re.search(rf"k{nodeNumber}[n,d,l].*", colName)
                        if match:
                            self.nodeSensitivityDict[node] = match.group()
                            self.nodeSensitivityList.append(match.group())
                            break
                """only keep the required sensitivites"""
                self.HVTrafoNode = self.data.loc[self.data['Name'] == 'Trafo_HSMS', 'Loc_from'].values[0]
                for agent in self.flexAgents:
                    match = re.search(rf"k{self.HVTrafoNode[9:]}[n,d,l].*", agent.id)
                    if match:
                        self.nodeSensitivityDict[self.HVTrafoNode] = match.group()
                        if match.group() not in self.nodeSensitivityList:
                            self.nodeSensitivityList.append(match.group())
                        break
                """check if the node is already present in the dictionary, else add the agent to the sensitivity matrix"""
                self.agentsWithNewNodeList = []
                for agent in self.flexAgents:
                    agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agent.id).group(1)
                    if not agentNode in self.nodeSensitivityDict:
                        self.agentsWithNewNodeList.append(agent.id)
                        self.nodeSensitivityDict[agentNode] = agent.id
        else:
            with Pool(processes=self.numCPU) as pool:
                dfList = pool.map(self.readSensi, fileList)
            self.sensitivity = pd.concat(dfList, axis=0, ignore_index=False)
            self.sensitivity.to_pickle("../inputs/SensiDFNew.pkl")
        for agent in self.flexAgents:
            self.agentNodes.append('Standort_' + re.search("k(\d+)[n,d,l]" , agent.id).group(1))

    def readSensi(self, file):
        df = pd.read_csv(file, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                         encoding='unicode_escape')
        df.rename(columns={'Unnamed: 0': 'Name', 'Unnamed: 1': 'time_step'}, inplace=True)
        df.drop('Unnamed: 6176', axis=1, inplace=True)
        df = df.filter(['Name', 'time_step'] + self.agentsWithNewNodeList + self.nodeSensitivityList)
        """take only those lines present in the grid even if sensitivity matrix have extra"""
        df = df.loc[df['Name'].isin(self.data['Name']), :]
        df.set_index('Name', inplace=True)
        return df

    def getCurrentSensitivity(self, time, node=None):
        if node is None:
            """get only the columns corresponding to self.loadsAndGens for vectorization"""
            columns = self.nodeSensitivityList[:len(self.loadsAndGens.columns)]
            sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time + 1, columns]
            if len(sensitivity) == 0:
                """some sensitivity matrices are missing, use the previous ones instead"""
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time, columns]
            sensitivity.columns = self.loadsAndGens.columns
        elif isinstance(node, list):
            """get the sensitivities of all the agents"""
            columns = [self.nodeSensitivityDict[n] for n in node]
            sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time + 1, columns]
            if len(sensitivity) == 0:
                """some sensitivity matrices are missing, use the previous ones instead"""
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time, columns]
        else:
            sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time + 1,
                                           self.nodeSensitivityDict[node]]
            if len(sensitivity) == 0:
                """some sensitivity matrices are missing, use the previous ones instead"""
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time,
                                                   self.nodeSensitivityDict[node]]
        return sensitivity

    def getSensitivity(self, time):
        sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time + 1, :]
        if len(sensitivity) == 0:
            """some sensitivity matrices are missing, use the previous ones instead"""
            sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time, :]
        return sensitivity

    def isCongested(self, spotBidDF):
        self.powerFlowApprox(spotBidDF)
        ratedIMat = self.data.loc[:, 'I_rated_A'].to_numpy().reshape(-1, 1)
        self.congestionStatus.index = self.dailyTimes
        self.congestionStatus.loc[:, :] = self.loading.abs().loc[:, :] > ratedIMat.T
        if self.congestionStatus.any(axis=None):
            self.reqdFlexTimes = self.dailyTimes[self.congestionStatus.any(axis='columns').values]
            return True
        else:
            return False

    def powerFlowApprox(self, spotBidDF):
        linesAndNodes = self.sensitivity.index[self.sensitivity['time_step'] == 1].values
        self.loading = pd.DataFrame(np.full((self.dailyFlexTime, len(linesAndNodes)), 0.0),
                                    columns=linesAndNodes, index=self.dailyTimes)
        for time in self.dailyTimes:
            totalLoad = 0
            totalGen = 0
            """add the contribution from households and other static agents to the lines"""
            sensitivity = self.getCurrentSensitivity(time).to_numpy()
            qty = self.loadsAndGens.loc[time, :].to_numpy().reshape((1,-1))
            self.loading.loc[time, :] = qty @ sensitivity.T
            if np.sum(qty) >= 0:
                totalLoad += np.sum(qty)
            else:
                totalGen += np.sum(qty)
            """add the flex agent contribution to the lines"""
            qty = spotBidDF.loc[time, :].to_numpy().reshape((1,-1))
            sensitivity = self.getCurrentSensitivity(time, self.agentNodes).to_numpy()
            self.loading.loc[time, :] += (qty @ sensitivity.T).ravel()
            if np.sum(qty) >= 0:
                totalLoad += np.sum(qty)
            else:
                totalGen += np.sum(qty)
            """include remaining power flow from HV transformer"""
            sensitivity = self.getCurrentSensitivity(time, self.HVTrafoNode)
            self.loading.loc[time, :] += sensitivity.values * (totalLoad + totalGen)

    def getStatus(self):
        return self.congestionStatus

    def endDay(self):
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)
