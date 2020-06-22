import numpy as np
import pandas as pd

from FlexAgent import FlexAgent
from AgentNeuralNet import AgentNeuralNet


class DSM(FlexAgent):
    def __init__(self, id, location=[0, 0], maxPower = 0, marginalCost = 0):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "Demand Side Management"
        self.scheduledLoad = None

        self.lowSpotBidLimit = 0
        self.highSpotBidLimit = 1
        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount 
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1

        # TODO define a max and min loading timeseries for using flexibility

        self.reset()

    def reset(self):
        super().reset()

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1.2, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1.2, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(1, 5, size=self.dailySpotTime)

        self.scheduledLoad = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                                'load': np.random.uniform(0, self.maxPower, size=self.spotTimePeriod)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def isAboveScheduled(self, time, power):
        """
        check whether the spot bid is above the scheduled load
        If False, returns by how much amount
        """
        # if self.scheduledLoad.query('time == time')['load'] <= power:
        if self.scheduledLoad.loc[time, 'load'] <= power:
            return True, None
        else:
            return False, self.scheduledLoad.loc[time, 'load'] - power

    def makeSpotBid(self):
        # explicitly bounding low limit to 0 for spot
        # TODO check if this approach is ok
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()
        """
        Make sure that the spot bid is always above the scheduled load
        """
        for i, qty in  zip(self.dailySpotBid['time'].values, self.dailySpotBid['qty_bid'].values):
            """DSM bids above scheduled load sop that in flex market, may be they can sell these in 
            case of constraints in those periods"""
            # TODO should it be always above schedule? do we need to include below schedule case also?
            possible, constraintAmount = self.isAboveScheduled(time=i, power=qty)
            if possible:
                pass
            else:
                # increase bid at least by constraintAmount
                self.spotBid.loc[i, 'qty_bid'] += np.random.uniform(constraintAmount,
                                                                    self.maxPower-self.spotBid.loc[i, 'qty_bid'])
            assert self.spotBid.loc[i, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'

    def spotMarketEnd(self):
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)

    def spotMarketReward(self, time, qty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_spot'] = self.dailySpotBid.loc[time, 'MCP'] * -qty

        # TODO what about the times it is not dispatched? Penalize? but already rewards are negative!

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

    def loadingScenario(self):
        # TODO define hourly loading of the the DSM
        pass
