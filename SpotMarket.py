import numpy as np
import pandas as pd


class SpotMarket:
    def __init__(self, spotTimePeriod = 8760, dailySpotTime = 24):
        self.participants = []
        self.bids = {}
        self.spotTimePeriod = spotTimePeriod
        self.dailySpotTime = dailySpotTime
        self.day = 0
        self.dailyTimes = None
        # hourly market clearing price
        self.MCP = None
        self.reset()

    def reset(self):
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        # hourly market clearing price
        self.MCP = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                      'price': np.zeros(self.spotTimePeriod, dtype=float)})

    def addParticipants(self, participants):
        # participants is a list of FlexAgents
        self.participants.extend(participants)

    def importPrice(self):
        # TODO get the market clearing price
        pass

    def collectBids(self):
        for participant in self.participants:
            participant.makeSpotBid()
            assert len(participant.getSpotBid().index) == self.dailySpotTime, 'Length of spot market and bid not equal'
            self.bids[participant.getID()] = participant.getSpotBid()

    def sendDispatch(self):
        for participant in self.participants:
            self.bids[participant.getID()].loc[:, 'dispatched'] = True

            participant.dailySpotBid.loc[:, 'MCP'] = self.MCP['price']
            participant.spotMarketEnd()

    def endDay(self):
        for participant in self.participants:
            participant.day += 1
            participant.dailyTimes = np.arange(participant.day * participant.dailySpotTime,
                                               (participant.day + 1) * participant.dailySpotTime)
        # update day for spot market object also
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
