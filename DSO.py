from FlexMarket import FlexMarket
from Grid import Grid


class DSO:
    def __init__(self, grid):
        self.flexAgents = []
        self.flexMarket = FlexMarket()
        self.grid = grid

    def checkCongestion(self):
        # TODO check the congestion efficiently
        status = self.grid.isCongested(self.flexMarket.dailyTimes)
        return status

    def addflexAgents(self, agents):
        # agents is a list of FlexAgents
        self.flexAgents.extend(agents)
        self.flexMarket.addParticipants(self.flexAgents)

    def askFlexibility(self):
        for agent in self.flexAgents:
            agent.spotState = False
        self.flexMarket.collectBids(self.grid)

    def optFlex(self, grid):
        # TODO choose the optimum flexibility wrt cost, qty and sensitivities
        self.flexMarket.sendDispatch()

    def endDay(self):
        for agent in self.flexAgents:
            agent.spotState = True
        self.flexMarket.endDay()
