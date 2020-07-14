import numpy as np
import pandas as pd
import os
import xlrd

from FlexAgent import FlexAgent


class PVG(FlexAgent):
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, voltageLevel= 0, marginalCost = 0, feedInPremium = 64):

        super().__init__(id=id, location=location, minPower = minPower, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "PV Generation"
        self.feedInPremium = feedInPremium
        """Assumption: the timeseries data contains -ve qty (generation)"""

        self.reset()

        self.importTimeseries()

    def reset(self):
        super().reset()
        self.importTimeseries()

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)

        self.flexBid.loc[:, 'price'] = np.full(self.flexTimePeriod, self.feedInPremium, dtype=float)
        self.flexBid.loc[:, 'qty_bid'] = np.full(self.flexTimePeriod, -self.maxPower, dtype=float)

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def importTimeseries(self):
        # get the timeseries data from grits
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "data\PV_timeseries_example.xlsx"
        abs_file_path = os.path.join(script_dir, rel_path)
        ts = pd.read_excel(abs_file_path, sheet_name=0)
        # TODO try with the real timeseries
        self.spotBid.loc[:, 'qty_bid'] = ts['generation']

    def makeSpotBid(self):
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()
        # self.spotBid.loc[self.spotBid['qty_bid'] > self.maxPower, 'qty_bid'] = self.maxPower

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