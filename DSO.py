from FlexMarket import FlexMarket
from Grid import Grid
import pandas as pd
import numpy as np

class DSO:
    def __init__(self, grid):
        self.flexAgents = []
        self.flexMarket = FlexMarket()
        self.grid = grid
        self.bids = {}

    def checkCongestion(self):
        # TODO check the congestion efficiently
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
                                          columns=self.flexAgents)
            for time in self.grid.reqdFlexTimes:
                self.bids[time] = pd.DataFrame(data={'qty': np.full(nBids, 0),
                                                     'price': np.full(nBids, 0),
                                                     'accepted': np.full(nBids, False)})
                """congested lines/ nodes at this particular time"""
                congested = self.grid.data.loc[self.grid.congestionStatus.loc[time, :].values, 'Name'].values
                """"""
                reqdFlexI_A = self.grid.data.loc[self.grid.data['Name'].isin(congested), 'I_rated_A'] - self.grid.loading.loc[time, congested]
                impact = pd.DataFrame(np.full((nBids, len(congested)), 0), columns=congested)
                for i, (agentID, flexbid) in enumerate(self.flexMarket.bids.items()):
                    """get the sensitivity of this agent on the congested lines"""
                    sensitivity = self.grid.sensitivity.loc[congested, agentID].values
                    self.bids[time].loc[i, ['qty', 'price']] = flexbid.loc[time, ['qty_bid', 'price']]
                    self.bids[time].rename(index={i: agentID})
                    """determine the impact of flexbid on the lines"""
                    # TODO should I negate the sensitivityMat?
                    impact.loc[i, :] = self.bids[time].loc[i, 'qty'] * sensitivity
                    impact.rename(index={i: agentID})
                for i, congestedLine in enumerate(impact):
                    """assuming if reqdFlexI_A is -3A(reduce 3A), everything less than -3A eg., -4A, -6A can be options """
                    options = (impact.loc[:, congestedLine] < reqdFlexI_A[i]).values
                    if options.count(True) > 1:
                        """compare the price to decide"""
                        # TODO minimum or maximum price depends upon whether DSO has to pay or gets money from flexagents
                        acceptedAgentID = self.bids[time].loc[options, 'price'].idxmin()
                        dispatchStatus.loc[time, acceptedAgentID] = True
                        self.bids[time].loc[acceptedAgentID, 'accepted'] = True
                    elif options.count(True) == 1:
                        acceptedAgentID = self.bids[time].index.values[options]
                        dispatchStatus.loc[time, acceptedAgentID] = True
                        self.bids[time].loc[acceptedAgentID, 'accepted'] = True
        self.flexMarket.sendDispatch(dispatchStatus)

    def endDay(self):
        for agent in self.flexAgents:
            agent.spotState = True
        self.flexMarket.endDay()
        self.grid.endDay()
