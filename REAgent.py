import numpy as np
import pandas as pd
import os
import xlrd
from FlexAgent import FlexAgent


class REAgent(FlexAgent):
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, voltageLevel= 0, marginalCost = 0,
                 feedInPremium = None, genSeries=None):
        super().__init__(id=id, location=location, minPower=minPower, maxPower=maxPower, marginalCost=marginalCost)
        self.feedInPremium = feedInPremium
        """Assumption: the timeseries data contains -ve qty (generation)"""
        self.reset(genSeries)

    def reset(self, genSeries):
        super().reset()
        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.spotBid.loc[:, 'qty_bid'] = genSeries
        self.flexBid.loc[:, 'qty_bid'] = genSeries
        self.flexBid.loc[:, 'price'] = np.full(self.flexTimePeriod, self.feedInPremium, dtype=float)

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def makeSpotBid(self):
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()

    def spotMarketEnd(self):
        super().spotMarketEnd()
        self.spotMarketReward()

    def spotMarketReward(self):
        """
        Renewables receive Feed-in Premium form the market
        Fixed  FIP is implemented here
        """
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[:, 'reward_spot'] = (self.dailySpotBid.loc[:, 'MCP'] -
                                                       self.marginalCost + self.feedInPremium) * \
                                                      -self.dailySpotBid.loc[:, 'qty_bid']
        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        super().makeFlexBid(reqdFlexTimes)

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

    def flexMarketReward(self, time, qty, price):
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)