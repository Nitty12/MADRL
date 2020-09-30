from FlexMarket import FlexMarket
from Grid import Grid
import pandas as pd
import numpy as np
import re
import util
import time as t


class DSO:
    def __init__(self, grid):
        self.flexAgents = []
        self.flexMarket = FlexMarket()
        self.grid = grid

    def checkCongestion(self, spotBidDF):
        status = self.grid.isCongested(spotBidDF)
        return status

    def addflexAgents(self, agents):
        # agents is a list of FlexAgents
        self.flexAgents.extend(agents)
        self.grid.flexAgents.extend(agents)
        self.grid.importLoadsAndSensi()
        self.flexMarket.addParticipants(self.flexAgents)

    def askFlexibility(self):
        for agent in self.flexAgents:
            agent.spotState = False
        self.flexMarket.collectBids(self.grid.reqdFlexTimes)

    def optFlex(self):
        """choose the optimum flexibility wrt cost, qty and sensitivities"""
        bids = None
        nBids = len(self.flexMarket.bids)
        dispatchStatus = None
        if len(self.grid.reqdFlexTimes) > 0:
            dispatchStatus = pd.DataFrame(np.full((len(self.grid.reqdFlexTimes), len(self.flexAgents)), False),
                                          columns=[agent.id for agent in self.flexAgents],
                                          index=self.grid.reqdFlexTimes)
            for time in self.grid.reqdFlexTimes:
                bids, currentSensitivity, sortedReqdFlexI_A, \
                congested, impact, sensitivityFlexAgents = self.getOptFlexData(nBids, time)
                sensitivityFlexAgents, bids, impact = self.buildSensiAndImpactDF(currentSensitivity, congested,
                                                                                 sensitivityFlexAgents, bids, impact, time)
                for i, congestedLine in enumerate(impact):
                    acceptedAgentID = None
                    """assuming if reqdFlexI_A is -3A(reduce 3A), everything less than -3A eg., -4A, -6A can be options
                        if its positive, everything more than reqdFlexI_A can be options"""
                    if sortedReqdFlexI_A[i] > 0:
                        options = (impact.loc[:, congestedLine] > sortedReqdFlexI_A[i]).values
                    else:
                        options = (impact.loc[:, congestedLine] < sortedReqdFlexI_A[i]).values

                    if list(options).count(True) > 1:
                        """compare the price to decide"""
                        acceptedAgentID = bids.loc[options, 'price'].idxmin()
                    elif list(options).count(True) == 1:
                        acceptedAgentID = bids.index.values[options]
                    else:
                        acceptedAgentID = self.chooseMultipleFlexOptions(sortedReqdFlexI_A, impact, congestedLine, i)

                    if acceptedAgentID is not None:
                        dispatchStatus.loc[time, acceptedAgentID] = True
                        bids.loc[acceptedAgentID, 'accepted'] = True
                        sortedReqdFlexI_A = self.updateRemainingLines(bids, congested, sensitivityFlexAgents,
                                                                  sortedReqdFlexI_A, acceptedAgentID, i, time)
        self.flexMarket.sendDispatch(dispatchStatus)

    def getOptFlexData(self, nBids, time):
        bids = pd.DataFrame(data={'qty_bid': np.full(nBids, 0),
                                       'price': np.full(nBids, 0),
                                       'accepted': np.full(nBids, False)})
        currentSensitivity = self.grid.getSensitivity(time)
        """congested lines/ nodes at this particular time"""
        congested = self.grid.data.loc[self.grid.congestionStatus.loc[time, :].values, 'Name'].values
        reqdFlexI_A = self.grid.data.loc[self.grid.data['Name'].isin(congested), 'I_rated_A'].values - \
                      np.abs(self.grid.loading.loc[time, congested].values)
        """sorting the required flexibility in reverse order to solve the biggest one first"""
        sortedReqdFlexI_A = reqdFlexI_A[np.abs(reqdFlexI_A).argsort()][::-1]
        congested = congested[np.abs(reqdFlexI_A).argsort()][::-1]

        impact = pd.DataFrame(np.full((nBids, len(congested)), 0), columns=congested)
        """stores the sensi of the agents to the congested lines"""
        sensitivityFlexAgents = pd.DataFrame(np.full((nBids, len(congested)), 0.0),
                                             columns=congested)
        return bids, currentSensitivity, sortedReqdFlexI_A, congested, impact, sensitivityFlexAgents

    def buildSensiAndImpactDF(self, currentSensitivity, congested, sensitivityFlexAgents, bids, impact, time):
        for i, (agentID, flexbid) in enumerate(self.flexMarket.bids.items()):
            agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentID).group(1)
            """get the sensitivity of this agent on the congested lines"""
            sensitivity = currentSensitivity.loc[congested, self.grid.nodeSensitivityDict[agentNode]].values
            sensitivityFlexAgents.loc[i, :] = sensitivity
            sensitivityFlexAgents.rename(index={i: agentID}, inplace=True)
            bids.loc[i, ['qty_bid', 'price']] = flexbid.loc[time, ['qty_bid', 'price']]
            bids.rename(index={i: agentID}, inplace=True)
            """determine the impact of flexbid on the lines"""
            impact.loc[i, :] = bids.loc[agentID, 'qty_bid'] * sensitivity
            impact.rename(index={i: agentID}, inplace=True)
            return sensitivityFlexAgents, bids, impact

    def chooseMultipleFlexOptions(self, sortedReqdFlexI_A, impact, congestedLine, i):
        """No single agent can reduce congestion, select multiple agents"""
        acceptedAgentID = None
        if sortedReqdFlexI_A[i] > 0:
            """selecting agents having atleast 0.5A positive effect to prevent very small impacts to be selected"""
            positiveImpacts = impact.loc[impact[congestedLine] > 0.5, congestedLine]
            if not len(positiveImpacts) == 0:
                if positiveImpacts.sum() < sortedReqdFlexI_A[i]:
                    """select all the positive impacting agents"""
                    acceptedAgentID = positiveImpacts.index.values
                else:
                    """arrange according to the impacts and select the minimum number of agents which 
                    eliminates congestion"""
                    minI_A = 0
                    for cumSum in positiveImpacts.sort_values(ascending=False).cumsum().values:
                        if cumSum > sortedReqdFlexI_A[i]:
                            minI_A = cumSum
                            break
                    indexMask = (positiveImpacts.sort_values(ascending=False).cumsum() <= minI_A).values
                    acceptedAgentID = positiveImpacts.index[indexMask].values
        else:
            positiveImpacts = impact.loc[impact[congestedLine] < -0.5, congestedLine]
            if not len(positiveImpacts) == 0:
                if positiveImpacts.sum() > sortedReqdFlexI_A[i]:
                    """Sum of all positively impacting flexbids is even not sufficient, so select all these bids"""
                    acceptedAgentID = positiveImpacts.index.values
                else:
                    maxI_A = 0
                    for cumSum in positiveImpacts.sort_values(ascending=False).cumsum().values:
                        if cumSum < sortedReqdFlexI_A[i]:
                            maxI_A = cumSum
                            break
                    indexMask = (positiveImpacts.sort_values(ascending=True).cumsum() >= maxI_A).values
                    acceptedAgentID = positiveImpacts.index[indexMask].values
        return acceptedAgentID

    def updateRemainingLines(self, bids, congested, sensitivityFlexAgents, sortedReqdFlexI_A, acceptedAgentID, i, time):
        """change the loading of the remaining lines according to the dispatched flexibilities"""
        if i < len(congested) - 1:
            for remainingCongestedLine in congested[i + 1:]:
                try:
                    acceptedBids = bids.loc[acceptedAgentID, 'qty_bid'].values.reshape((-1, 1))
                    sensi = sensitivityFlexAgents.loc[acceptedAgentID, remainingCongestedLine].values
                    self.grid.loading.loc[time, remainingCongestedLine] += acceptedBids.T @ sensi
                except:
                    acceptedBids = bids.loc[acceptedAgentID, 'qty_bid']
                    sensi = sensitivityFlexAgents.loc[acceptedAgentID, remainingCongestedLine]
                    self.grid.loading.loc[time, remainingCongestedLine] += acceptedBids * sensi
                sortedReqdFlexI_A[i + 1] = self.grid.data.loc[self.grid.data['Name'] == remainingCongestedLine,
                                                              'I_rated_A'] - np.abs(self.grid.loading.loc[time,
                                                                                                          remainingCongestedLine])
        return sortedReqdFlexI_A

    def endDay(self):
        for agent in self.flexAgents:
            agent.spotState = True
        self.flexMarket.endDay()
        self.grid.endDay()
