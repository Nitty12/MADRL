import numpy as np
import pandas as pd


class Grid:
    def __init__(self, TimePeriod=8760, numNodes=0, numLines=0):
        self.TimePeriod = TimePeriod
        self.numNodes = numNodes
        self.numLines = numLines
        self.nodes = []
        self.lines = []
        self.lineLoading = pd.DataFrame(np.full((self.numLines, 2), 0.0), columns=['limit', 'current'])
        self.status = pd.DataFrame(data={'time': np.arange(self.TimePeriod),
                                         'congestion': np.full(self.TimePeriod, False)})
        self.congestedLines = pd.DataFrame(np.full((self.TimePeriod, self.numLines), False))
        self.congestedNodes = pd.DataFrame(np.full((self.TimePeriod, self.numNodes), False))

    def importGrid(self):
        # TODO import the load flow sensitivities
        for col in self.congestedLines:
            self.status.loc[:, 'congestion'] |= self.congestedLines[col]

    def addLinesAndnodes(self):
        self.congestedLines.columns = self.lines
        self.congestedNodes.columns = self.nodes
        newIndex = {}
        for i, line in enumerate(self.lines):
            newIndex[i] = line
        self.lineLoading.rename(index=newIndex, inplace=True)

    def loadFlowApprox(self):
        # TODO set values in congestedLines and congestedNodes
        pass

    def getStatus(self):
        return self.status
