import numpy as np
import pandas as pd
import os
import re


class Grid:
    def __init__(self, TimePeriod=8760, numNodes=0, numLines=0):
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
        self.householdLoad = None
        """contains the list of households in each node"""
        self.householdNodeDict = None
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
        self.getHouseholdLoad()
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

    def getHouseholdLoad(self):
        path = os.getcwd()
        datapath = os.path.join(path, "../inputs/ang_Kunden_HH1_nnf_corrected_1h.csv")
        loadingSeriesColumns = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                    encoding='unicode_escape', nrows=0)
        loadingSeriesColumns.drop('NNF', axis=1, inplace=True)
        columnNames1 = list(loadingSeriesColumns)
        loadingSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                    encoding='unicode_escape', dtype=float)
        loadingSeries.drop('NNF', axis=1, inplace=True)
        loadingSeries.columns = columnNames1

        datapath = os.path.join(path, "../inputs/ang_Kunden_HH2_nnf_corrected_1h.csv")
        loadingSeriesColumns = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape', nrows=0)
        loadingSeriesColumns.drop('NNF', axis=1, inplace=True)
        columnNames2 = list(loadingSeriesColumns)
        loadingSeries2 = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                     encoding='unicode_escape', dtype=float)
        loadingSeries2.drop('NNF', axis=1, inplace=True)
        loadingSeries2.columns = columnNames2
        loadingSeries = pd.concat([loadingSeries, loadingSeries2], axis=1, sort=False)
        # TODO delete efficiently may be use multiprocess pool etc
        del loadingSeries2

        """get the list of nodes in which the household loads are connected"""
        nodes = []
        for name in columnNames1+columnNames2:
            n = re.search("(Standort_[0-9]+)", name)
            if not n.group() in nodes:
                nodes.append(n.group())

        """get the list of households in each node
            add the loads of houses connected to the same node and only store the result"""
        self.householdLoad = pd.DataFrame(columns=nodes)
        householdNodeDict = {}
        for node in nodes:
            householdNodeDict[node] = []
            for name in columnNames1+columnNames2:
                match = re.search(rf".*{node}.*", name)
                if match:
                    householdNodeDict[node].append(match.group())
            self.householdLoad[node] = loadingSeries.loc[:, householdNodeDict[node]].sum(axis=1)

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
                for node in self.householdLoad:
                    nodeNumber = node[9:]
                    for colName in df:
                        match = re.search(rf"k{nodeNumber}[n,d].*", colName)
                        if match:
                            self.nodeSensitivityDict[node] = match.group()
                            nodeSensitivityList.append(match.group())
                            break
                """only keep the required sensitivites"""
                self.HVTrafoNode = self.data.loc[self.data['Name'] == 'Trafo_HSMS', 'Loc_from'].values[0]
                for agent in self.flexAgents:
                    match = re.search(rf"k{self.HVTrafoNode[9:]}[n,d].*", agent.id)
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
            """add the household loads to the lines"""
            for node in self.householdLoad:
                sensitivity = self.sensitivity.loc[self.sensitivity['time_step'] == time+1,
                                                      self.nodeSensitivityDict[node]]
                # TODO should I negate the sensitivityMat?
                load = self.householdLoad.loc[time, node]
                self.loading.loc[time % self.dailyFlexTime, :] += sensitivity.values * load
                totalLoad += load
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
