import numpy as np
import pandas as pd
from FlexAgent import FlexAgent


class DSM(FlexAgent):
    def __init__(self, id, location=[0, 0], maxPower = 0, marginalCost = 0, scheduledLoad=None, startDay=0, endDay=365):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost,
                         startDay=startDay, endDay=endDay)
        self.type = "Demand Side Management"
        self.scheduledLoad = pd.DataFrame(data={'time': self.totalTimes,
                                                'load': scheduledLoad}, index=self.totalTimes)
        self.baseLoad = pd.DataFrame(data={'time': self.totalTimes,
                                           'load': 0.2*scheduledLoad}, index=self.totalTimes)
        """spotBidMultiplier of DSM agent is used to redistribute the difference (P_total - P_base)
            of each hour : To incorporate the constraints
                                    if for each hour P_total > Pmax --> Negative rewards
                                    if total of 24 hr P_total > scheduledLoad --> Negative rewards
                                    if total of 24 hr P_total < scheduledLoad --> Negative rewards"""
        self.lowSpotBidLimit = 0
        self.highSpotBidLimit = 2
        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount 
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        """if status is True, the time is non flexible time"""
        self.nonFlexibleTimes = pd.DataFrame(data={'time': self.totalTimes,
                                                   'status': np.full(self.spotTimePeriod, False)},
                                             index=self.totalTimes)
        # TODO initialize the non available times

        self.reset(startDay)

    def reset(self, startDay=0):
        super().reset(startDay)
        """changing spot qty bid from maxPower to scheduled load"""
        self.spotBid.loc[:, 'qty_bid'] = self.scheduledLoad.loc[:, 'load']
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]
        self.flexBid.loc[:, 'qty_bid'] = self.spotBid.loc[:, 'qty_bid']
        self.dailyFlexBid = self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)]

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def makeSpotBid(self):
        # TODO check if this approach is ok
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        self.spotBidMultiplier = self.getSpotBidMultiplier()
        dailyTotalLoad = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes), 'qty_bid']
        dailyBaseLoad = self.baseLoad.loc[self.baseLoad['time'].isin(self.dailyTimes), 'load']
        dailyNonFlexTimes = self.nonFlexibleTimes.loc[self.nonFlexibleTimes['time'].isin(self.dailyTimes), 'status']
        """modifies the above baseLoad load in spot market to optimize with the prices
            also taking care of the nonFlexibleTimes"""
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes), 'qty_bid'] = \
            dailyBaseLoad + (dailyTotalLoad - dailyBaseLoad)*self.spotBidMultiplier
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]
        # TODO is this approach or giving high penalty for qty changed in nonFlexTimes better?
        self.dailySpotBid.loc[dailyNonFlexTimes, 'qty_bid'] = self.scheduledLoad.loc[self.scheduledLoad['time'].isin
                                                                                     (self.dailyTimes), 'load']

    def spotMarketEnd(self):
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)

    def spotMarketReward(self, time, qty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_spot'] = self.dailySpotBid.loc[time, 'MCP'] * -qty

        """penalize for spot bid constraint violations"""
        totalScheduledLoad = self.scheduledLoad.loc[self.scheduledLoad['time'].isin(self.dailyTimes), 'load'].sum()
        totalSpotBid = self.dailySpotBid.loc[:, 'qty_bid'].sum()
        if totalSpotBid > totalScheduledLoad:
            self.dailyRewardTable.loc[self.dailyTimes[0], 'reward_spot'] += (totalSpotBid - totalScheduledLoad) \
                                                                            * self.penaltyViolation
        else:
            self.dailyRewardTable.loc[self.dailyTimes[0], 'reward_spot'] += (totalScheduledLoad - totalSpotBid) \
                                                                            * self.penaltyViolation
        totalExcessBid = self.dailySpotBid.loc[self.dailySpotBid.loc[:, 'qty_bid'] > self.maxPower, 'qty_bid'].sum()
        self.dailyRewardTable.loc[self.dailyTimes[0], 'reward_spot'] += totalExcessBid * self.penaltyViolation

        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        """initialize to the (current spot bid quantities - P_base) the qty that can be shifted"""
        dailyBaseLoad = self.baseLoad.loc[self.baseLoad['time'].isin(self.dailyTimes), 'load']
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'qty_bid'] = self.dailySpotBid.loc[:, 'qty_bid'] \
                                                                                  - dailyBaseLoad
        super().makeFlexBid(reqdFlexTimes)
        """if the flexbid is greater than (maxPower-spot dispatched qty), 
        its not valid """
        for i, qty in zip(self.dailyFlexBid['time'].values, self.dailyFlexBid['qty_bid'].values):
            highLimit = self.maxPower
            if self.dailySpotBid.loc[i, 'dispatched']:
                highLimit = self.maxPower - self.dailySpotBid.loc[i, 'qty_bid']

            if not qty <= highLimit:
                # TODO bid is not valid, penalize the agent?
                self.dailyFlexBid.loc[i, 'qty_bid'] = 0
                self.dailyFlexBid.loc[i, 'price'] = 0
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

    def flexMarketReward(self, time, qty, price):
        # Either way, if dispatched, the DSO pays the agents
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * qty.abs()
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)
