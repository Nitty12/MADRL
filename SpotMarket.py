import numpy as np
import pandas as pd
import os
import util
import time

class SpotMarket:
    def __init__(self, spotTimePeriod = 8760, dailySpotTime = 24):
        self.participants = []
        self.bids = {}
        self.bidsDF = None
        self.spotTimePeriod = spotTimePeriod
        self.dailySpotTime = dailySpotTime
        self.day = 0
        self.dailyTimes = None
        # hourly market clearing price
        self.MCP = None
        self.reset()

    def reset(self, startDay=0):
        self.day = startDay
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        self.nextDayTimes = np.arange((self.day + 1) * self.dailySpotTime, (self.day +2) * self.dailySpotTime)
        # hourly market clearing price
        self.MCP = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                      'price': np.zeros(self.spotTimePeriod, dtype=float)})
        self.importPrice()

    def addParticipants(self, participants):
        # participants is a list of FlexAgents
        self.participants.extend(participants)
        agentNameList = [agent.id for agent in self.participants]
        self.bidsDF = pd.DataFrame(np.full((self.dailySpotTime, len(self.participants)), 0.0), columns=agentNameList,
                                   index=self.dailyTimes)

    def importPrice(self):
        path = os.getcwd()
        datapath = os.path.join(path, "../inputs/Day-ahead Prices_201901010000-202001010000_neu.csv")
        self.MCP = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0,
                          usecols=['MTU (CET)', 'Day-ahead Price [EUR/MWh]'], error_bad_lines=False)
        self.MCP.columns = ['time', 'price']

    def collectBids(self):
        self.bidsDF.loc[:, :] = 0.0
        self.bidsDF.index=self.dailyTimes
        for participant in self.participants:
            participant.makeSpotBid()
            assert len(participant.getSpotBid().index) == self.dailySpotTime, 'Length of spot market and bid not equal'
            spotBid = participant.getSpotBid()
            self.bids[participant.getID()] = spotBid
            self.bidsDF.loc[:, participant.id] = spotBid.loc[:, 'qty_bid']
            participant.setMCP(self.MCP.loc[self.nextDayTimes, 'price'].values)

    def sendDispatch(self):
        for participant in self.participants:
            self.bids[participant.getID()].loc[:, 'dispatched'] = True
            participant.dailySpotBid.loc[:, 'MCP'] = self.MCP.loc[participant.dailyTimes,'price']
            participant.spotMarketEnd()

    def endDay(self):
        for participant in self.participants:
            participant.day += 1
            participant.dailyTimes = np.arange(participant.day * participant.dailySpotTime,
                                               (participant.day + 1) * participant.dailySpotTime)
        # update day for spot market object also
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        self.nextDayTimes = np.arange((self.day + 1) * self.dailySpotTime, (self.day +2) * self.dailySpotTime)
