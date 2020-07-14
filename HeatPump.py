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
        self.storageLevelBeforeSpot = None  # how much energy is stored in the reservoir
        self.storageLevelAfterSpot = None
        # self.heatPrice = 50
        self.maxHeatLoad = maxHeatLoad
        if minStorageLevel is None:
            self.minStorageLevel = 0.2 * self.maxStorageLevel

        self.ambTemp = None

        self.scheduledLoad = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                                'load': scheduledLoad,
                                                'loaded': np.full(self.spotTimePeriod, False)})
        self.energyTable = None

        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount 
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1

        self.reset()

    def reset(self):
        super().reset()
        self.storageLevelBeforeSpot = self.maxStorageLevel
        self.storageLevelAfterSpot = self.maxStorageLevel

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(0, 1, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(1, 5, size=self.dailySpotTime)
        self.energyTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'before_spot': np.full(self.spotTimePeriod, self.storageLevelBeforeSpot,
                                                                     dtype=float),
                                              'dispatch_spot': np.full(self.spotTimePeriod, False),
                                              'after_spot': np.full(self.spotTimePeriod, self.storageLevelAfterSpot,
                                                                    dtype=float),
                                              'realised_spot': np.full(self.spotTimePeriod, True),
                                              'before_flex': np.full(self.flexTimePeriod, self.storageLevelBeforeSpot,
                                                                     dtype=float),
                                              'dispatch_flex': np.full(self.spotTimePeriod, False),
                                              'after_flex': np.full(self.flexTimePeriod, self.storageLevelBeforeSpot,
                                                                    dtype=float)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def updateEnergyTable(self, index, status, value):
        if not index == self.spotTimePeriod:
            self.energyTable.loc[index, status] = value

    def canLoad(self, load, time, status, index):
        # TODO change checking minstoragelevel to next actual load
        energy = load * time
        if status == 'before_spot':
            if self.storageLevelBeforeSpot - energy >= self.minStorageLevel:
                return True
            else:
                return False
        elif status == 'after_spot':
            if self.storageLevelAfterSpot - energy >= self.minStorageLevel:
                return True
            else:
                return False
        elif status == 'before_flex':
            if self.energyTable.loc[index, status] - energy >= self.minStorageLevel:
                return True
            else:
                return False
        elif status == 'after_flex':
            if self.energyTable.loc[index, status] - energy >= self.minStorageLevel:
                return True
            else:
                return False

    def load(self, load, time, status, index):
        energy = load * time
        if status == 'before_spot':
            self.storageLevelBeforeSpot -= energy
            self.updateEnergyTable(index, status, self.storageLevelBeforeSpot)
        elif status == 'after_spot':
            self.storageLevelAfterSpot -= energy
            self.updateEnergyTable(index, status, self.storageLevelAfterSpot)
        elif status == 'before_flex':
            self.updateEnergyTable(index, status, self.energyTable.loc[index-1, 'before_flex'] - energy)

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
            if self.storageLevelBeforeSpot + energy > self.maxStorageLevel:
                return False, self.storageLevelBeforeSpot + energy - self.maxStorageLevel
            else:
                return True, None
        elif status == 'before_flex':
            if self.spotBid.loc[index, 'dispatched']:
                spotDispatchedQty = self.spotBid.loc[index, 'qty_bid']
            else:
                spotDispatchedQty = 0

            if spotDispatchedQty + self.energyTable.loc[index, status] + energy > self.maxStorageLevel:
                return False, spotDispatchedQty + self.energyTable.loc[index, status] + energy - self.maxStorageLevel
            else:
                return True, None

    def store(self, power, time, status, index):
        energy = power * time
        if status == 'before_spot':
            self.storageLevelBeforeSpot += energy
            self.updateEnergyTable(index, status, self.storageLevelBeforeSpot)
        elif status == 'before_flex':
            value = self.energyTable.loc[index, status] + energy
            self.updateEnergyTable(index, status, value)

    def checkLoading(self, status, index):
        load = self.scheduledLoad.loc[index, 'load']
        possible = self.canLoad(load, self.spotTimeInterval, status, index)
        if possible:
            """
            energyTable holds energy at the start of the specified time
            so, a load in current time affects the energy at the next time
            """
            self.load(load=load, time=self.spotTimeInterval, status=status, index=index+1)
            if status == 'after_spot':
                self.scheduledLoad.loc[index, 'loaded'] = True
            return True
        else:
            # print("WARNING: Heat pump {0} cannot be loaded at index {1} {2}!".format(self.id, index, status))
            # TODO what can be done here
            return False

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

        status = 'after_spot'
        self.storageLevelAfterSpot = None
        penalizeTimes = []
        for time, energy in zip(self.dailySpotBid['time'].values,
                                self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes),
                                                     'before_spot'].values):
            if self.storageLevelAfterSpot is None:
                self.storageLevelAfterSpot = energy
                self.energyTable.loc[time, status] = self.storageLevelAfterSpot
            """
            checkLoading updates the energyTable with the load
                if cannot be loaded --> high negative rewards"""
            loaded = self.checkLoading(status=status, index=time)
            if not loaded:
                penalizeTimes.append(time)

            if time in spotDispatchedTimes:
                dispatchedEnergy = spotDispatchedQty.loc[time] * self.spotTimeInterval
            else:
                dispatchedEnergy = 0.0
            """
            energyTable holds energy at the start of the specified time
            so, a load in current time affects the energy at the next time
            """
            if not time + 1 == self.spotTimePeriod:
                if not self.minStorageLevel <= (self.storageLevelAfterSpot + dispatchedEnergy) \
                        <= self.maxStorageLevel:
                    """
                    if constraints are violated after the spot clearance, the agent cannot dispatch
                    """
                    self.energyTable.loc[time, 'realised_spot'] = False
                    dispatchedEnergy = 0.0
                    if not penalizeTimes[-1] == time:
                        penalizeTimes.append(time)
                self.energyTable.loc[time + 1, status] = self.storageLevelAfterSpot + dispatchedEnergy
            self.storageLevelAfterSpot += dispatchedEnergy

        # realised = self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'realised_spot'].values
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values, penalizeTimes)
        """set the storage levels"""
        self.storageLevelBeforeSpot = self.storageLevelAfterSpot

    def spotMarketReward(self, time, qty, penalizeTimes):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        """Here negative of qty is used for reward because generation is negative qty"""
        self.dailyRewardTable.loc[time, 'reward_spot'] = (self.dailySpotBid.loc[time, 'MCP'] - self.marginalCost) * -qty
        self.dailyRewardTable.loc[penalizeTimes, 'reward_spot'] = self.penaltyViolation
        # """
        # For the unrealised dispatch, a negative reward is given wrt to MCP for maxPower qty"""
        # self.dailyRewardTable.loc[~realisedStatus, 'reward_spot'] = self.dailySpotBid.loc[~realisedStatus, 'MCP'] \
        #                                                        * -np.abs(self.maxPower)

        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        status = 'before_flex'
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        super().makeFlexBid(reqdFlexTimes)
        self.makeBid(self.dailyFlexBid, status)
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def makeBid(self, dailyBid, status):
        for t, qty, i in zip(dailyBid['time'].values, dailyBid['qty_bid'].values, range(24)):
            """
            checkLoading updates the energyTable with the load
            """
            self.checkLoading(status=status, index=t)
            if qty > 0:
                possible, constraintAmount = self.canStore(power=qty, time=self.spotTimeInterval, status=status, index=t)
                if possible:
                    self.store(power=qty, time=self.spotTimeInterval, status=status, index=t+1)
                else:
                    """storing not possible: we are not modifying the bid, but the energy bought is of 'no use'
                                             ie., cannot be stored but has to be paid - almost like a penalty"""
                    self.store(power=0, time=self.spotTimeInterval, status=status, index=t+1)
                    # TODO check other possible options here
            else:
                """In case of Flexbid: check whether it can reduce this particular amount from spot bid
                    if possible, consider it as if it was a load"""
                possible = self.canLoad(load=qty, time=self.spotTimeInterval, status=status, index=t)
                if possible:
                    self.load(load=qty, time=self.spotTimeInterval, status=status, index=t+1)
                else:
                    """if reduction is not possible, modify the bid because it is not realisable"""
                    self.dailyFlexBid.loc[i, 'qty_bid'] = 0
            assert -self.maxPower <= dailyBid.loc[i, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_flex'] = \
            self.dailyFlexBid.loc[:, 'dispatched']
        penalizeTimes = []
        for time, dispatched in zip(self.dailyFlexBid['time'].values,
                                    self.dailyFlexBid.loc[:, 'dispatched'].values):
            if not time+1 == self.spotTimePeriod:
                if dispatched:
                    # TODO check if this is correct!
                    # self.energyTable.loc[time+1, 'after_flex'] = self.energyTable.loc[time+1, 'before_flex']
                    flexDispatchedEnergy = self.dailyFlexBid.loc[self.dailyFlexBid['time']==time, 'qty_bid'] \
                                           * self.spotTimeInterval
                    self.energyTable.loc[time+1, 'after_flex'] = self.energyTable.loc[time+1, 'after_spot'] \
                                                                 + flexDispatchedEnergy
                    """
                    checkLoading updates the energyTable with the load
                        if cannot be loaded --> high negative rewards"""
                    loaded = self.checkLoading(status='after_flex', index=time)
                    if not loaded:
                        penalizeTimes.append(time)
                else:
                    """No change in energy"""
                    self.energyTable.loc[time+1, 'after_flex'] = self.energyTable.loc[time, 'after_flex']

        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice, penalizeTimes)
        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
        the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1)*self.dailySpotTime
        self.energyTable.loc[nextDayFirstTime, ['before_spot', 'after_spot', 'before_flex', 'after_flex']] = \
            self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.storageLevelBeforeSpot = self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.storageLevelAfterSpot= self.energyTable.loc[nextDayFirstTime, 'after_flex']

    def flexMarketReward(self, time, qty, price, penalizeTimes):
        self.dailyRewardTable.loc[penalizeTimes, 'reward_flex'] = self.penaltyViolation
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)


