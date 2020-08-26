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
        """Assumption: the timeseries data contains -ve qty (generation)
            In spot bids, it can reduce the generation so, spot limits are (0 to 1)*generation forecast
            In flex bids, it can again reduce the generation or increase the generation upto the 
                possible limit(generation forecast-spot dispatch qty)
                but, the possibility of increase is not considered because it can only increase congestion
                so, flex bid limits are (-1 to 0)*spot dispatched qty
            """
        self.genSeries = genSeries
        self.reset()

    def reset(self):
        super().reset()
        self.spotBid.loc[:, 'qty_bid'] = self.genSeries
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]
        self.flexBid.loc[:, 'qty_bid'] = self.genSeries
        self.flexBid.loc[:, 'price'] = np.full(self.flexTimePeriod, self.feedInPremium, dtype=float)
        self.dailyFlexBid = self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)]

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
        """copy spot bids so that the flexbid multiplier is applied on this instead of maxPower"""
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'qty_bid'] = self.dailySpotBid.loc[:, 'qty_bid']
        super().makeFlexBid(reqdFlexTimes)

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

    def flexMarketReward(self, time, qty, price):
        # Here the qty will be positive as it is as though it is taking back the spot generated qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * qty
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)