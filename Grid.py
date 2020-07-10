import numpy as np
import pandas as pd
import os

class Grid:
    def __init__(self, TimePeriod=8760, numNodes=0, numLines=0):
        self.TimePeriod = TimePeriod
        self.numNodes = numNodes
        self.numLines = numLines
        self.nodes = []
        self.lines = []
        self.data = None
        self.loading = None
        self.congestedLines = None
        self.congestedNodes = None
        # self.status = pd.DataFrame(data={'time': np.arange(self.TimePeriod),
        #                                  'congestion': np.full(self.TimePeriod, False)})
        """for testing"""
        self.status = pd.DataFrame(data={'time': np.arange(self.TimePeriod),
                                         'congestion': np.random.choice([True, False], size=(self.TimePeriod, ))})
        self.importGrid()

    def importGrid(self):
        """get the grid data from the CSV"""
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "data\CBCO_Results_.csv"
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

        self.loading = pd.DataFrame(np.full((self.numNodes + self.numLines, 3), 0.0),
                                    columns=['Name', 'I_rated_A', 'I_A'])
        self.loading.loc[:, 'Name'] = self.data.loc[:, 'Name']
        self.congestedLines = pd.DataFrame(np.full((self.TimePeriod, self.numLines), False))
        self.congestedLines.columns = self.lines
        self.congestedNodes = pd.DataFrame(np.full((self.TimePeriod, self.numNodes), False))
        self.congestedNodes.columns = self.nodes

        # TODO import the load flow sensitivities

    def checkCongestion(self, time):
        """for testing"""
        self.congestedLines.loc[time, :] = np.random.choice([True, False], size=(len(time), self.numLines))
        self.congestedNodes.loc[time, :] = np.random.choice([True, False], size=(len(time), self.numNodes))
        for col_name in self.congestedLines:
            self.status.loc[time, 'congestion'] |= self.congestedLines[time, col_name]
        return np.any(self.status.loc[time, 'congestion'].values)

    def loadFlowApprox(self):
        # TODO set values in congestedLines and congestedNodes
        pass

    def getStatus(self):
        return self.status
