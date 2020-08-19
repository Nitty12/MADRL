from FlexMarket import FlexMarket
from Grid import Grid
import pandas as pd
import numpy as np
import re


class DSO:
    def __init__(self, grid):
        self.flexAgents = []
        self.flexMarket = FlexMarket()
        self.grid = grid
        self.bids = {}

    def checkCongestion(self):
        status = self.grid.isCongested()
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
        # TODO choose the optimum flexibility wrt cost, qty and sensitivities
        self.bids = {}
        nBids = len(self.flexMarket.bids)
        dispatchStatus = None
        if len(self.grid.reqdFlexTimes) > 0:
            dispatchStatus = pd.DataFrame(np.full((len(self.grid.reqdFlexTimes), len(self.flexAgents)), False),
                                          columns=[agent.id for agent in self.flexAgents],
                                          index=self.grid.reqdFlexTimes)
            for time in self.grid.reqdFlexTimes:
                self.bids[time] = pd.DataFrame(data={'qty_bid': np.full(nBids, 0),
                                                     'price': np.full(nBids, 0),
                                                     'accepted': np.full(nBids, False)})
                currentSensitivity = self.grid.sensitivity.loc[self.grid.sensitivity['time_step']==time+1, :]
                """congested lines/ nodes at this particular time"""
                congested = self.grid.data.loc[self.grid.congestionStatus.loc[time, :].values, 'Name'].values
                """"""
                reqdFlexI_A = self.grid.data.loc[self.grid.data['Name'].isin(congested), 'I_rated_A'].values - \
                              self.grid.loading.loc[time, congested].values
                impact = pd.DataFrame(np.full((nBids, len(congested)), 0), columns=congested)
                for i, (agentID, flexbid) in enumerate(self.flexMarket.bids.items()):
                    agentNode = 'Standort_' + re.search("k(\d+)[n,d,l]", agentID).group(1)
                    """get the sensitivity of this agent on the congested lines"""
                    sensitivity = currentSensitivity.loc[congested, self.grid.nodeSensitivityDict[agentNode]].values
                    self.bids[time].loc[i, ['qty_bid', 'price']] = flexbid.loc[time, ['qty_bid', 'price']]
                    self.bids[time].rename(index={i: agentID}, inplace=True)
                    """determine the impact of flexbid on the lines"""
                    # TODO should I negate the sensitivityMat?
                    impact.loc[i, :] = self.bids[time].loc[agentID, 'qty_bid'] * sensitivity
                    impact.rename(index={i: agentID}, inplace=True)
                for i, congestedLine in enumerate(impact):
                    """assuming if reqdFlexI_A is -3A(reduce 3A), everything less than -3A eg., -4A, -6A can be options
                        if its positive, everything more than reqdFlexI_A can be options"""
                    if reqdFlexI_A[i] > 0:
                        options = (impact.loc[:, congestedLine] > reqdFlexI_A[i]).values
                    else:
                        options = (impact.loc[:, congestedLine] < reqdFlexI_A[i]).values
                    if list(options).count(True) > 1:
                        """compare the price to decide"""
                        # TODO minimum or maximum price depends upon whether DSO has to pay or gets money from flexagents
                        acceptedAgentID = self.bids[time].loc[options, 'price'].idxmin()
                    elif list(options).count(True) == 1:
                        acceptedAgentID = self.bids[time].index.values[options]
                    else:
                        """No single agent can reduce congestion, select multiple agents"""
                        if reqdFlexI_A[i] > 0:
                            positiveImpacts = impact.loc[impact[congestedLine] > 0, congestedLine]
                            if positiveImpacts.sum() < reqdFlexI_A[i]:
                                """select all the positive impacting agents"""
                                acceptedAgentID = positiveImpacts.index.values
                            else:
                                """arrange according to the impacts and select the minimum number of agents which 
                                eliminates congestion"""
                                minI_A = 0
                                for cumSum in positiveImpacts.sort_values(ascending=False).cumsum().values:
                                    if cumSum > reqdFlexI_A[i]:
                                        minI_A = cumSum
                                        break
                                indexMask = (positiveImpacts.sort_values(ascending=False).cumsum() <= minI_A).values
                                acceptedAgentID = positiveImpacts.index[indexMask].values
                        else:
                            positiveImpacts = impact.loc[impact[congestedLine] < 0, congestedLine]
                            if positiveImpacts.sum() > reqdFlexI_A[i]:
                                acceptedAgentID = positiveImpacts.index.values
                            else:
                                maxI_A = 0
                                for cumSum in positiveImpacts.sort_values(ascending=False).cumsum().values:
                                    if cumSum < reqdFlexI_A[i]:
                                        maxI_A = cumSum
                                        break
                                indexMask = (positiveImpacts.sort_values(ascending=True).cumsum() >= maxI_A).values
                                acceptedAgentID = positiveImpacts.index[indexMask].values
                    dispatchStatus.loc[time, acceptedAgentID] = True
                    self.bids[time].loc[acceptedAgentID, 'accepted'] = True
        self.flexMarket.sendDispatch(dispatchStatus)

    def endDay(self):
        for agent in self.flexAgents:
            agent.spotState = True
        self.flexMarket.endDay()
        self.grid.endDay()
