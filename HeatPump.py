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
        self.storageLevelAfterSpot = None  # how much energy is stored in the reservoir
        self.storageLevelAfterFlex = None
        self.storageLevelBeforeFlex = None
        # self.heatPrice = 50
        self.maxHeatLoad = maxHeatLoad
        if minStorageLevel is None:
            self.minStorageLevel = 0.2 * self.maxStorageLevel

        self.ambTemp = None

        self.scheduledLoad = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                                'load': scheduledLoad,
                                                'loaded': np.full(self.spotTimePeriod, False)})
        self.energyTable = None
        self.spotChangedEnergy = None
        self.flexChangedEnergy = None

        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount 
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        self.penalizeTimes = []
        self.reset()

    def reset(self):
        super().reset()
        self.storageLevelAfterSpot = 0
        self.storageLevelBeforeFlex = 0
        self.storageLevelAfterFlex = 0
        self.spotChangedEnergy = 0
        self.flexChangedEnergy = 0

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(1, 5, size=self.dailySpotTime)
        self.energyTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'after_spot': np.full(self.spotTimePeriod, self.storageLevelAfterSpot,
                                                                    dtype=float),
                                              'before_flex': np.full(self.flexTimePeriod, self.storageLevelBeforeFlex,
                                                                    dtype=float),
                                              'after_flex': np.full(self.flexTimePeriod, self.storageLevelAfterFlex,
                                                                    dtype=float)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def updateEnergyTable(self, index, status, value):
        if not index == self.spotTimePeriod:
            self.energyTable.loc[index, status] = value

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
            if self.storageLevelAfterSpot + energy > self.maxStorageLevel:
                return False, self.storageLevelAfterSpot + energy - self.maxStorageLevel
            else:
                return True, None
        elif status == 'before_flex':
            """check whether additional this energy can be stored
                calculate how much energy was stored in this time in spot and update storageLevelBeforeFlex"""
            if self.storageLevelBeforeFlex + self.spotChangedEnergy > self.maxStorageLevel:
                return False, self.storageLevelBeforeFlex + self.spotChangedEnergy - self.maxStorageLevel
            else:
                return True, None
        elif status == 'after_flex':
            if self.storageLevelAfterFlex + self.flexChangedEnergy > self.maxStorageLevel:
                return False, self.storageLevelAfterFlex + self.flexChangedEnergy - self.maxStorageLevel
            else:
                return True, None

    def store(self, power, time, status, index):
        energy = power * time
        if status == 'before_spot':
            self.storageLevelAfterSpot += energy
            self.updateEnergyTable(index, status, self.storageLevelAfterSpot)
        elif status == 'before_flex':
            self.storageLevelBeforeFlex = self.storageLevelBeforeFlex + self.spotChangedEnergy
            self.updateEnergyTable(index, status, self.storageLevelBeforeFlex)
        elif status == 'after_flex':
            self.storageLevelAfterFlex = self.storageLevelAfterFlex + self.flexChangedEnergy
            self.updateEnergyTable(index, status, self.storageLevelAfterFlex)

    def checkLoading(self, status, index):
        load = self.scheduledLoad.loc[index, 'load']
        possible = self.canLoad(load, self.spotTimeInterval, status, index)
        if possible:
            """
            energyTable holds energy at the start of the specified time
            so, a load in current time affects the energy at the next time
            """
            self.load(load=load, time=self.spotTimeInterval, status=status, index=index+1)
            if status == 'after_flex':
                self.scheduledLoad.loc[index, 'loaded'] = True
            return True
        else:
            # print("WARNING: Heat pump {0} cannot be loaded at index {1} {2}!".format(self.id, index, status))
            # TODO what can be done here
            return False

    def canLoad(self, load, time, status, index):
        # TODO change checking minstoragelevel to next actual load
        energy = load * time
        if status == 'before_spot':
            if self.storageLevelAfterSpot - energy >= self.minStorageLevel:
                return True
            else:
                return False
        elif status == 'before_flex':
            if self.storageLevelBeforeFlex + self.spotChangedEnergy >= self.minStorageLevel:
                return True
            else:
                return False
        elif status == 'after_flex':
            if self.storageLevelAfterFlex + self.flexChangedEnergy >= self.minStorageLevel:
                return True
            else:
                return False

    def load(self, load, time, status, index):
        energy = load * time
        if status == 'before_spot':
            self.storageLevelAfterSpot -= energy
            self.updateEnergyTable(index, status, self.storageLevelAfterSpot)
        elif status == 'before_flex':
            self.storageLevelBeforeFlex = self.storageLevelBeforeFlex + self.spotChangedEnergy
            self.updateEnergyTable(index, status, self.storageLevelBeforeFlex)
        elif status == 'after_flex':
            self.storageLevelAfterFlex = self.storageLevelAfterFlex + self.flexChangedEnergy
            self.updateEnergyTable(index, status, self.storageLevelAfterFlex)

    def makeSpotBid(self):
        status = 'before_spot'
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()
        self.makeBid(self.dailySpotBid, status)
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)] = self.dailySpotBid

    def spotMarketEnd(self):
        self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_spot'] = \
            self.dailySpotBid.loc[:, 'dispatched']
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)

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
        super().makeFlexBid(reqdFlexTimes)
        self.penalizeTimes = []
        self.makeBid(self.dailyFlexBid, status)
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def makeBid(self, dailyBid, status):
        for t, qty, i in zip(dailyBid['time'].values, dailyBid['qty_bid'].values, range(24)):
            if status == 'before_flex':
                """amount of energy changed in the spot dispatch used to update the energy table for before_flex times"""
                self.spotChangedEnergy = self.energyTable.loc[t+1, 'after_spot'] - self.energyTable.loc[t, 'after_spot']

            """checkLoading updates the energyTable with the load
                    if cannot be loaded --> high negative rewards"""
            loaded = self.checkLoading(status=status, index=t)
            if status == 'before_spot':
                """only penalize spot bid while making bid bcoz for flex bid we dont know which of the bids gets 
                    accepted, so penalize after dispatch"""
                self.penalizeTimes = []
                if not loaded:
                    self.penalizeTimes.append(t)

            if qty > 0:
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
                    # TODO what to do here?
                    pass
            assert -self.maxPower <= dailyBid.loc[i, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_flex'] = \
            self.dailyFlexBid.loc[:, 'dispatched']
        self.penalizeTimes = []
        for time, qty, dispatched in zip(self.dailyFlexBid['time'].values,
                                         self.dailyFlexBid['qty_bid'].values,
                                         self.dailyFlexBid.loc[:, 'dispatched'].values):
            """amount of energy changed in the spot and flex dispatch used to update the energy table for after_flex 
            times """
            self.flexChangedEnergy = self.energyTable.loc[time + 1, 'before_flex'] - self.energyTable.loc[time, 'before_flex']
            self.spotChangedEnergy = self.energyTable.loc[time + 1, 'after_spot'] - self.energyTable.loc[time, 'after_spot']

            """checkLoading updates the energyTable with the load
                    if cannot be loaded --> high negative rewards"""
            loaded = self.checkLoading(status='after_flex', index=time)
            if not loaded:
                self.penalizeTimes.append(time)

            if not time+1 == self.spotTimePeriod:
                """
                checkLoading updates the energyTable with the load
                    if cannot be loaded --> high negative rewards"""
                loaded = self.checkLoading(status='after_flex', index=time)
                if not loaded:
                    self.penalizeTimes.append(time)
                if dispatched:
                    if qty > 0:
                        possible, _ = self.canStore(power=qty, time=self.spotTimeInterval, status='after_flex', index=time)
                        if possible:
                            self.store(power=qty, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                        else:
                            if not self.penalizeTimes[-1] == time:
                                self.penalizeTimes.append(time)
                    else:
                        """consider negative bid as load"""
                        possible = self.canLoad(load=-qty, time=self.spotTimeInterval, status='after_flex', index=time)
                        if possible:
                            self.load(load=-qty, time=self.spotTimeInterval, status='after_flex', index=time + 1)
                        else:
                            if not self.penalizeTimes[-1] == time:
                                self.penalizeTimes.append(time)
                else:
                    self.storageLevelAfterFlex = self.storageLevelAfterFlex + self.spotChangedEnergy
                    self.updateEnergyTable(time+1, 'after_flex', self.storageLevelAfterFlex)

        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
        the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1)*self.dailySpotTime
        self.energyTable.loc[nextDayFirstTime, ['before_spot', 'before_flex', 'after_flex']] = \
            self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.storageLevelAfterSpot = self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.storageLevelBeforeFlex = self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.storageLevelAfterFlex = self.energyTable.loc[nextDayFirstTime, 'after_flex']

    def flexMarketReward(self, time, qty, price):
        self.dailyRewardTable.loc[self.penalizeTimes, 'reward_flex'] = self.penaltyViolation
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)


