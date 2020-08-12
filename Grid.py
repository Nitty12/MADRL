import numpy as np
import pandas as pd
import os
import re


class Grid:
    def __init__(self,numAgents, loadingSeriesHP, chargingSeriesEV, genSeriesPV, genSeriesWind, loadingSeriesDSM,
                 TimePeriod=8760, numNodes=0, numLines=0):
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
        self.data = None
        self.loading = None
        self.HVTrafoNode = None
        # self.congestedLines = None
        # self.congestedNodes = None
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
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "../inputs/CBCO_Results_.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.data = pd.read_csv(abs_file_path, sep=';', comment='#', header=0, skiprows=1, error_bad_lines=False)
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

        for name in self.data['Name']:
            if name.startswith(('Trafo', 'Knoten')):
                self.nodes.append(name)
            elif name.startswith('L'):
                self.lines.append(name)
        self.numNodes = len(self.nodes)
        self.numLines = len(self.lines)

        self.loading = pd.DataFrame(np.full((self.dailyFlexTime, self.numNodes + self.numLines), 0.0),
                                    columns=list(self.data.loc[:, 'Name']))
        self.congestionStatus = pd.DataFrame(np.full((self.dailyFlexTime, self.numNodes + self.numLines), False),
                                             columns=list(self.data.loc[:, 'Name']))

    def importLoadsAndSensi(self):
        """get the household load data from the CSV"""
        self.getLoadsAndGens()
        """load the sensitivity matrix and concatenate"""
        self.loadSensitivityMatrix()

    def isCongested(self):
        self.powerFlowApprox()
        ratedIMat = self.data.loc[:, 'I_rated_A'].to_numpy().reshape(-1, 1)
        self.congestionStatus.loc[:, :] = self.loading.loc[:, :] > ratedIMat.T
        if self.congestionStatus.any(axis=None):
            self.reqdFlexTimes = self.dailyTimes[self.congestionStatus.any(axis='columns').values]
            return True
        else:
            return False

    def getLoadsAndGens(self):
        path = os.getcwd()
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
        # TODO delete efficiently may be use multiprocess pool etc
        del loadingSeriesHH2

        columnNames = columnNamesHH1 + columnNamesHH2 + list(self.loadingSeriesHP.columns) + list(self.chargingSeriesEV.columns) \
                      + list(self.loadingSeriesDSM.columns) + list(self.genSeriesPV.columns) + list(self.genSeriesWind.columns)

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
        dfList = []
        agentList = []
        nodeSensitivityList = []
        for file in fileList:
            df = pd.read_csv(file, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                             encoding='unicode_escape')
            df.rename(columns={'Unnamed: 0': 'Name', 'Unnamed: 1': 'time_step'}, inplace=True)
            df.drop('Unnamed: 6672', axis=1, inplace=True)
            if not dfList:
                """gets the first flexagent name connected to a particular node to get the sensitivity to use for 
                    power flow approximation"""
                for node in self.loadsAndGens:
                    nodeNumber = node[9:]
                    for colName in df:
                        match = re.search(rf"k{nodeNumber}[n,d,l].*", colName)
                        if match:
                            self.nodeSensitivityDict[node] = match.group()
                            nodeSensitivityList.append(match.group())
                            break
                """only keep the required sensitivites"""
                self.HVTrafoNode = self.data.loc[self.data['Name'] == 'Trafo_HSMS', 'Loc_from'].values[0]
                for agent in self.flexAgents:
                    match = re.search(rf"k{self.HVTrafoNode[9:]}[n,d,l].*", agent.id)
                    if match:
                        self.nodeSensitivityDict[self.HVTrafoNode] = match.group()
                        if match.group() not in nodeSensitivityList:
                            nodeSensitivityList.append(match.group())
                        break
                agentList = [agent.id for agent in self.flexAgents if agent.id not in nodeSensitivityList]
            df = df.filter(['Name', 'time_step'] + agentList + nodeSensitivityList)
            """take only those lines present in the grid even if sensitivity matrix have extra"""
            df = df.loc[df['Name'].isin(self.data['Name']), :]
            df.set_index('Name', inplace=True)
            dfList.append(df)
        self.sensitivity = pd.concat(dfList, axis=0, ignore_index=True)

    def powerFlowApprox(self):
        self.loading = pd.DataFrame(np.full((self.dailyFlexTime, self.numNodes + self.numLines), 0.0),
                                    columns=list(self.data.loc[:, 'Name']))
        for time in self.dailyTimes:
            totalLoad = 0
            totalGen = 0
            """add the contribution from households and other static agents to the lines"""
            for node in self.loadsAndGens:
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time+1,
                                                      self.nodeSensitivityDict[node]]
                # TODO should I negate the sensitivityMat?
                qty = self.loadsAndGens.loc[time, node]
                self.loading.loc[time % self.dailyFlexTime, :] += sensitivity.values * qty
                if qty >= 0:
                    totalLoad += qty
                else:
                    totalGen += qty
            """add the flex agent contribution to the lines"""
            for agent in self.flexAgents:
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time+1,
                                                      agent.id]
                # TODO should I negate the sensitivityMat?
                qty = agent.dailySpotBid.loc[time, 'qty_bid']
                self.loading.loc[time % self.dailyFlexTime, :] += sensitivity.values * qty
                if qty >= 0:
                    totalLoad += qty
                else:
                    totalGen += qty
            """include remaining power flow from HV transformer"""
            sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time + 1,
                                               self.nodeSensitivityDict[self.HVTrafoNode]]
            self.loading.loc[time % self.dailyFlexTime, :] += sensitivity.values * (totalLoad - totalGen)

    def getStatus(self):
        return self.congestionStatus

    def endDay(self):
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)
