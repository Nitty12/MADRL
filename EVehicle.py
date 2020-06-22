import numpy as np
import pandas as pd

from FlexAgent import FlexAgent


class EVehicle(FlexAgent):
    '''
        Represents a VPP of electric vehicles
        the EMS of the VPP can provide:
            +ve flexibility: STOP charging
            -ve flexibility: START charging
    '''

    def __init__(self, id, location=[0, 0], maxPower = 0, marginalCost = 0,
                 nVehicles = 0, maxCapacity = 0, efficiency = 1.0):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "E-vehicle VPP"
        self.nVehicles = nVehicles
        self.maxCapacity = self.nVehicles * maxCapacity  # capacity in MWh
        self.maxPower = self.nVehicles * maxPower
        self.remainingEnergy = None
        self.efficiency = efficiency
        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount (eg, to stop/reduce charging)
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        self.reset()

    def reset(self):
        super().reset()
        self.remainingEnergy = self.maxCapacity

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1.2, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1.2, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(1, 5, size=self.dailySpotTime)


    def printInfo(self):
        super().printInfo()
        print("Type: {}\n".format(self.type))

    def importTimeseries(self):
        # get the timeseries data from grits
        path = ''
        ts = pd.read_csv(path)
        # TODO
        self.spotBid.loc[:, 'qty_bid'] = ts

    def makeSpotBid(self):
        # explicitly bounding low limit to 0 for spot
        # TODO check if this approach is ok
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        self.spotBid.loc[:, 'qty_bid'] *= self.spotBidMultiplier
        # TODO

    def spotMarketEnd(self):
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)

    def spotMarketReward(self, time, qty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_spot'] = self.dailySpotBid.loc[time, 'MCP'] * -qty
        # TODO what about the times it is not dispatched?
        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        super().makeFlexBid(reqdFlexTimes)

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice = super().flexMarketEnd()
        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)



