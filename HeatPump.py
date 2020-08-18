import numpy as np
import pandas as pd
from FlexAgent import FlexAgent


class HeatPump(FlexAgent):
    def __init__(self, id, location=[0, 0], maxPower=0, marginalCost=0,
                 maxStorageLevel=0, maxHeatLoad=0, minStorageLevel=None, scheduledLoad=None):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "Heat Pump"
        self.maxStorageLevel = maxStorageLevel  # Storage capacity in MWh
        """
        How much energy is stored in the reservoir after the spot market bids
        and after the spot market dispatch
        used to update the energyTable
        """
        self.storageLevel = None  # how much energy is stored in the reservoir
        # self.heatPrice = 50
        self.maxHeatLoad = maxHeatLoad
        if minStorageLevel is None:
            self.minStorageLevel = 0.2 * self.maxStorageLevel

        self.ambTemp = None

        self.scheduledLoad = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                                'load': scheduledLoad,
                                                'loaded': np.full(self.spotTimePeriod, False)})
        self.energyTable = None
        self.spotChangedEnergy = 0
        self.flexChangedEnergy = 0

        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount 
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        self.penalizeTimes = []
        self.reset()

    def reset(self):
        super().reset()
        self.storageLevel = self.minStorageLevel
        self.spotChangedEnergy = 0
        self.flexChangedEnergy = 0

        self.energyTable = pd.DataFrame(data={'time': np.arange(self.dailySpotTime+1),
                                              'after_spot': np.full(self.dailySpotTime+1, self.storageLevel,
                                                                    dtype=float),
                                              'before_flex': np.full(self.dailySpotTime+1, self.storageLevel,
                                                                    dtype=float),
                                              'after_flex': np.full(self.dailySpotTime+1, self.storageLevel,
                                                                    dtype=float)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def updateEnergyTable(self, index, status, value):
        if not index == self.spotTimePeriod:
            self.energyTable.loc[self.energyTable['time'] == index, status] = value

    def canStore(self, power, time, status, index):
        """
        check whether storing energy is possible
        time in hrs and power in MW (+ve)
        If False, returns by how much amount
        """
        if power > self.maxPower:
            power = self.maxPower

        energy = power * time
        if status == 'before_spot':
            if self.storageLevel + energy > self.maxStorageLevel:
                return False, self.storageLevel + energy - self.maxStorageLevel
            else:
                return True, None
        elif status == 'before_flex':
            """check whether additional this energy can be stored
                calculate how much energy was stored in this time in spot and update storageLevelBeforeFlex"""
            if self.storageLevel + self.spotChangedEnergy + energy > self.maxStorageLevel:
                return False, self.storageLevel + self.spotChangedEnergy + energy - self.maxStorageLevel
            else:
                return True, None
        elif status == 'after_flex':
            if self.storageLevel + self.flexChangedEnergy > self.maxStorageLevel:
                return False, self.storageLevel + self.flexChangedEnergy - self.maxStorageLevel
            else:
                return True, None

    def store(self, power, time, status, index):
        energy = power * time
        if status == 'before_spot':
            self.storageLevel += energy
            self.updateEnergyTable(index, 'after_spot', self.storageLevel)
        elif status == 'before_flex':
            self.storageLevel = self.storageLevel + self.spotChangedEnergy + energy
            self.updateEnergyTable(index, status, self.storageLevel)
        elif status == 'after_flex':
            self.storageLevel = self.storageLevel + self.flexChangedEnergy
            self.updateEnergyTable(index, status, self.storageLevel)

    def checkLoading(self, status, index):
        load = self.scheduledLoad.loc[index, 'load']
        if load == 0:
            return True
        possible = self.canLoad(load, self.spotTimeInterval, status, index)
        if possible:
            """only load before spot since others already take into account the loading via 
            spotChangedEnergy and flexChangedEnergy"""
            if status == 'before_spot':
                """
                energyTable holds energy at the start of the specified time
                so, a load in current time affects the energy at the next time
                """
                self.load(load=load, time=self.spotTimeInterval, status='after_spot', index=index+1)
            return True
        else:
            # print("WARNING: Heat pump {0} cannot be loaded at index {1} {2}!".format(self.id, index, status))
            # TODO what can be done here
            return False

    def canLoad(self, load, time, status, index):
        # TODO change checking minstoragelevel to next actual load
        energy = load * time
        if self.storageLevel - energy >= self.minStorageLevel:
            return True
        else:
            return False

    def load(self, load, time, status, index):
        energy = load * time
        self.storageLevel -= energy
        self.updateEnergyTable(index, status, self.storageLevel)

    def makeSpotBid(self):
        status = 'before_spot'
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        self.energyTable.loc[:, 'time'] = np.concatenate([self.dailyTimes, np.array([self.dailyTimes[-1]+1])])
        super().makeSpotBid()
        self.makeBid(self.dailySpotBid, status)
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)] = self.dailySpotBid

    def spotMarketEnd(self):
        # self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_spot'] = \
        #     self.dailySpotBid.loc[:, 'dispatched']
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)
        """change storage level to that of the starting time for this day to use in flex market"""
        self.storageLevel = self.energyTable.loc[self.energyTable['time'] == self.dailyTimes[0], 'after_spot'].values[0]

    def spotMarketReward(self, time, qty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        """Here negative of qty is used for reward because generation is negative qty"""
        self.dailyRewardTable.loc[time, 'reward_spot'] = (self.dailySpotBid.loc[time, 'MCP'] - self.marginalCost) * -qty
        """penalizing for not having enough energy stored for loading"""
        self.dailyRewardTable.loc[self.penalizeTimes, 'reward_spot'] = self.penaltyViolation
        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        status = 'before_flex'
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        """copy spot bids so that the flexbid multiplier is applied on this instead of maxPower"""
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'qty_bid'] = self.dailySpotBid.loc[:, 'qty_bid']
        super().makeFlexBid(reqdFlexTimes)
        self.penalizeTimes = []
        self.makeBid(self.dailyFlexBid, status)
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        # self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_flex'] = \
        #     self.dailyFlexBid.loc[:, 'dispatched']
        """change storage level to that of the starting time for this day to use in flex market"""
        self.storageLevel = self.energyTable.loc[self.energyTable['time'] == self.dailyTimes[0], 'after_spot'].values[0]
        self.penalizeTimes = []
        for time, qty, dispatched in zip(self.dailyFlexBid['time'].values,
                                         self.dailyFlexBid['qty_bid'].values,
                                         self.dailyFlexBid.loc[:, 'dispatched'].values):
            """amount of energy changed in the spot and flex dispatch used to update the energy table for after_flex 
            times """
            self.flexChangedEnergy = self.energyTable.loc[self.energyTable['time'] == time+1, 'before_flex'].values[0] -\
                                     self.energyTable.loc[self.energyTable['time'] == time, 'before_flex'].values[0]
            self.spotChangedEnergy = self.energyTable.loc[self.energyTable['time'] == time+1, 'after_spot'].values[0] -\
                                     self.energyTable.loc[self.energyTable['time'] == time, 'after_spot'].values[0]

            """checkLoading updates the energyTable with the load
                    if cannot be loaded --> high negative rewards"""

            loaded = self.checkLoading(status='after_flex', index=time)
            if not loaded:
                self.penalizeTimes.append(time)

            if not time+1 == self.spotTimePeriod:
                if dispatched:
                    if qty > 0:
                        possible, _ = self.canStore(power=qty, time=self.spotTimeInterval, status='after_flex', index=time)
                        if possible:
                            self.store(power=qty, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                        else:
                            """if the list is empty or if time has not been added"""
                            if not self.penalizeTimes or not self.penalizeTimes[-1] == time:
                                self.penalizeTimes.append(time)
                            self.store(power=0, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                    else:
                        """consider negative bid as load"""
                        possible = self.canLoad(load=-qty, time=self.spotTimeInterval, status='after_flex', index=time)
                        if possible:
                            self.load(load=-qty, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                        else:
                            if not self.penalizeTimes or not self.penalizeTimes[-1] == time:
                                self.penalizeTimes.append(time)
                            self.load(load=0, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                else:
                    self.storageLevel = self.storageLevel + self.spotChangedEnergy
                    self.updateEnergyTable(time+1, 'after_flex', self.storageLevel)

        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
        the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1)*self.dailySpotTime
        self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, ['after_spot', 'before_flex']] = \
            self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, 'after_flex']
        self.storageLevel = self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, 'after_flex'].values[0]
        """keep the energy at the end of last day as the starting energy"""
        self.energyTable.loc[0,  ['after_spot', 'before_flex', 'after_flex']] = self.storageLevel

    def flexMarketReward(self, time, qty, price):
        self.dailyRewardTable.loc[self.penalizeTimes, 'reward_flex'] = self.penaltyViolation
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)

    def makeBid(self, dailyBid, status):
        for t, qty, i in zip(dailyBid['time'].values, dailyBid['qty_bid'].values, range(24)):
            if status == 'before_flex':
                """amount of energy changed in the spot dispatch used to update the energy table for before_flex times"""
                self.spotChangedEnergy = self.energyTable.loc[self.energyTable['time'] == t+1, 'after_spot'].values[0] - \
                                         self.energyTable.loc[self.energyTable['time'] == t, 'after_spot'].values[0]

            """checkLoading updates the energyTable with the load
                    if cannot be loaded --> high negative rewards"""
            loaded = self.checkLoading(status=status, index=t)
            if status == 'before_spot':
                """only penalize spot bid while making bid bcoz for flex bid we dont know which of the bids gets 
                    accepted, so penalize after dispatch"""
                self.penalizeTimes = []
                if not loaded:
                    self.penalizeTimes.append(t)

            if qty >= 0:
                possible, _ = self.canStore(power=qty, time=self.spotTimeInterval, status=status, index=t)
                if possible:
                    self.store(power=qty, time=self.spotTimeInterval, status=status, index=t+1)
                else:
                    """storing not possible: we are not modifying the bid, but the energy bought is of 'no use'
                                             ie., cannot be stored but has to be paid - almost like a penalty"""
                    self.store(power=0, time=self.spotTimeInterval, status=status, index=t+1)
                    # TODO check other possible options here
            else:
                """In case of Flexbid: check whether it can reduce this particular amount from spot bid
                    if possible, consider it as if it was a load
                    reverse the sign as it is already negative"""
                possible = self.canLoad(load=-qty, time=self.spotTimeInterval, status=status, index=t)
                if possible:
                    self.load(load=-qty, time=self.spotTimeInterval, status=status, index=t+1)
                else:
                    # """if reduction is not possible, modify the bid because it is not realisable"""
                    # self.dailyFlexBid.loc[i, 'qty_bid'] = 0
                    self.load(load=0, time=self.spotTimeInterval, status=status, index=t + 1)
                    # TODO what to do here?
                    pass
            assert -self.maxPower <= dailyBid.loc[t, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'
