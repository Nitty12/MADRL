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
        # self.congestedLines = None
        # self.congestedNodes = None
        self.reqdFlexTimes = None
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
        self.nodeSensitivityDict = None
        self.importGrid()
        self.day = 0
        self.dailyFlexTime = 24
        self.dailyTimes = None
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
            if name.startswith('Trafo'):
                self.nodes.append(name)
            elif name.startswith('L'):
                self.lines.append(name)
        self.numNodes = len(self.nodes)
        self.numLines = len(self.lines)

        self.loading = pd.DataFrame(np.full((self.dailyFlexTime, self.numNodes + self.numLines), 0.0),
                                    columns=list(self.data.loc[:, 'Name']))
        self.congestionStatus = pd.DataFrame(np.full((self.dailyFlexTime, self.numNodes + self.numLines), False),
                                             columns=list(self.data.loc[:, 'Name']))
        """get the household load data from the CSV"""
        self.getHouseholdLoad()
        # TODO import the load flow sensitivities

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
        columnNames = list(loadingSeriesColumns)
        loadingSeries = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                    encoding='unicode_escape', dtype=float)
        loadingSeries.drop('NNF', axis=1, inplace=True)
        loadingSeries.columns = columnNames

        datapath = os.path.join(path, "../inputs/ang_Kunden_HH2_nnf_corrected_1h.csv")
        loadingSeriesColumns = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False,
                                     encoding='unicode_escape', nrows=0)
        loadingSeriesColumns.drop('NNF', axis=1, inplace=True)
        columnNames = list(loadingSeriesColumns)
        loadingSeries2 = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                     encoding='unicode_escape', dtype=float)
        loadingSeries2.drop('NNF', axis=1, inplace=True)
        loadingSeries2.columns = columnNames
        loadingSeries = pd.concat([loadingSeries, loadingSeries2], axis=1, sort=False)
        # TODO delete efficiently may be use multiprocess pool etc
        del loadingSeries2

        """get the list of nodes in which the household loads are connected"""
        nodes = []
        for name in columnNames:
            n = re.search("(Standort_[0-9]+)", name)
            if not n.group() in nodes:
                nodes.append(n.group())

        """get the list of households in each node
            add the loads of houses connected to the same node and only store the result"""
        self.householdLoad = pd.DataFrame(columns=nodes)
        householdNodeDict = {}
        for node in nodes:
            householdNodeDict[node] = []
            for name in columnNames:
                match = re.search(rf".*{node}.*", name)
                if match:
                    householdNodeDict[node].append(match.group())
            self.householdLoad[node] = loadingSeries.loc[:, householdNodeDict[node]].sum(axis=1)

    def loadSensitivityMatrix(self):
        path = os.getcwd()
        datapath = os.path.join(path, "../inputs/Sensitivities_Result_.csv")
        self.sensitivity = None
        self.sensitivity = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=2, error_bad_lines=False,
                                  encoding='unicode_escape')
        self.sensitivity.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
        self.sensitivity.drop('Unnamed: 1', axis=1, inplace=True)
        self.sensitivity.set_index('Name', inplace=True)
        self.sensitivity = self.sensitivity.apply(pd.to_numeric)
        """gets the first flexagent name connected to a particular node to get the sensitivity to use for 
                power flow approximation"""
        if not self.nodeSensitivityDict:
            for node in self.householdLoad:
                nodeNumber = node[9:]
                for colName in self.sensitivity:
                    match = re.search(rf"k{nodeNumber}[n,d].*", colName)
                    if match:
                        self.nodeSensitivityDict[node] = match.group()
                        break

    def powerFlowApprox(self):
        # TODO change the pathname for the correct loading at each time
        self.loadSensitivityMatrix()
        """add the household loads to the lines"""
        for node in self.householdLoad:
            sensitivityMat = self.sensitivity.loc[:, self.nodeSensitivityDict[node]].to_numpy().reshape(-1, 1)
            householdLoadMat = self.householdLoad.loc[self.dailyTimes, node].to_numpy().reshape(-1, 1)
            # TODO should I negate the sensitivityMat?
            self.loading.loc[:, :] += householdLoadMat @ sensitivityMat.T
        """add the flex agent contribution to the lines"""
        for agent in self.flexAgents:
            sensitivityMat = self.sensitivity.loc[:, agent.id].to_numpy().reshape(-1, 1)
            bidMat = agent.dailySpotBid.loc[: 'qty_bid'].to_numpy().reshape(-1, 1)
            # TODO should I negate the sensitivityMat?
            self.loading.loc[:, :] += bidMat @ sensitivityMat.T
        # TODO set values in congestedLines and congestedNodes

    def getStatus(self):
        return self.congestionStatus

    def endDay(self):
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)
