import numpy as np
import pandas as pd
from FlexAgent import FlexAgent


class BatStorage(FlexAgent):
    def __init__(self, id, location=[0, 0], minPower = 0, maxPower = 0, voltageLevel= 0, marginalCost=0,
                 maxCapacity=0, efficiency=1.0, SOC=1.0, minSOC=0.2):

        super().__init__(id=id, location=location, minPower = minPower, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "Battery Storage"
        self.maxCapacity = maxCapacity  # capacity in MWh
        """
        How much energy is left in the battery after the spot market bids
        and after the spot market dispatch
        used to update the energyTable
        """
        self.remainingEnergy = 0

        self.efficiency = efficiency
        self.SOC = SOC
        self.energyTable = None
        self.minSOC = minSOC
        self.minCapacity = self.minSOC * self.maxCapacity

        self.lowSpotBidLimit = -1
        self.highSpotBidLimit = 1
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        self.lowPriceLimit = 1
        self.highPriceLimit = 5

        self.penalizeTimes = []
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0

        self.reset()

    def reset(self):
        super().reset()
        self.remainingEnergy = 0
        self.SOC = 1.0
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0

        self.energyTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'after_spot': np.full(self.spotTimePeriod, self.remainingEnergy,
                                                                    dtype=float),
                                              'before_flex': np.full(self.flexTimePeriod, self.remainingEnergy,
                                                                     dtype=float),
                                              'after_flex': np.full(self.flexTimePeriod, self.remainingEnergy,
                                                                    dtype=float)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}".format(self.type))

    def canCharge(self, chargePower, chargeTime, status, index):
        """
        check whether charging is possible
        chargeTime in hrs and chargePower in MW (+ve)
        If False, returns by how much amount
        """
        if chargePower > self.maxPower:
            chargePower = self.maxPower

        energy = chargePower * chargeTime * self.efficiency
        if status == 'after_spot':
            if self.remainingEnergy + energy > self.maxCapacity:
                return False, self.remainingEnergy + energy - self.maxCapacity
            else:
                return True, None
        elif status == 'before_flex':
            """checks if in the current hour spot dispatched qty + remaining energy after flex bidding 
            leads to constraints violation or not"""
            if self.remainingEnergy + self.spotChangedEnergy + energy > self.maxCapacity:
                return False, self.remainingEnergy + self.spotChangedEnergy + energy - self.maxCapacity
            else:
                return True, None
        elif status == 'after_flex':
            if self.remainingEnergy + self.flexChangedEnergy > self.maxCapacity:
                return False, self.remainingEnergy + self.flexChangedEnergy - self.maxCapacity
            else:
                return True, None

    def canDischarge(self, dischargePower, dischargeTime, status, index):
        """
        check whether discharging is possible
        dischargeTime in hrs and dischargePower in MW (-ve)
        If False, returns by how much amount
        """
        if dischargePower < -self.maxPower:
            dischargePower = -self.maxPower

        # energy will be negative as dischargePower is negative
        energy = dischargePower * dischargeTime * self.efficiency
        if status == 'after_spot':
            if self.remainingEnergy + energy < self.minCapacity:
                return False, self.minCapacity - (self.remainingEnergy + energy)
            else:
                return True, None
        elif status == 'before_flex':
            """checks if in the current hour spot dispatched qty + remaining energy after flex bidding 
            leads to constraints violation or not"""
            if self.remainingEnergy + self.spotChangedEnergy + energy < self.minCapacity:
                return False, self.minCapacity - (self.remainingEnergy + self.spotChangedEnergy + energy)
            else:
                return True, None
        elif status == 'after_flex':
            if self.remainingEnergy + self.flexChangedEnergy < self.minCapacity:
                return False, self.minCapacity - (self.remainingEnergy + self.flexChangedEnergy)
            else:
                return True, None

    def changeSOC(self, qty, time, status, index):
        energy = qty * time * self.efficiency
        if status == 'after_spot':
            self.remainingEnergy += energy
        elif status == 'before_flex':
            self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy + energy
        elif status == 'after_flex':
            self.remainingEnergy = self.remainingEnergy + self.flexChangedEnergy
        if not index == self.spotTimePeriod:
            self.energyTable.loc[index, status] = self.remainingEnergy
        self.SOC = self.remainingEnergy / self.maxCapacity

    def makeSpotBid(self):
        status = 'after_spot'
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()
        self.makeBid(self.dailySpotBid, status)
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)] = self.dailySpotBid

    def spotMarketEnd(self):
        # self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_spot'] = \
        #     self.dailySpotBid.loc[:, 'dispatched']
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values)
        """change remaining energy to that of the starting time for this day to use in flex market"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time']==self.dailyTimes[0], 'after_spot'].values[0]

    def spotMarketReward(self, time, qty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_spot'] = (self.dailySpotBid.loc[time, 'MCP'] - self.marginalCost) * -qty
        self.dailyRewardTable.loc[self.penalizeTimes, 'reward_spot'] = self.penaltyViolation
        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] \
            = self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        status = 'before_flex'
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        super().makeFlexBid(reqdFlexTimes)
        self.makeBid(self.dailyFlexBid, status)
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice= super().flexMarketEnd()
        # self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_flex'] = \
        #     self.dailyFlexBid.loc[:, 'dispatched']
        """change remaining energy to that of the starting time for this day to update after_flex energy table"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time']==self.dailyTimes[0], 'after_spot'].values[0]
        self.penalizeTimes = []
        for time, qty, dispatched in zip(self.dailyFlexBid['time'].values,
                                         self.dailyFlexBid['qty_bid'].values,
                                         self.dailyFlexBid.loc[:, 'dispatched'].values):
            """amount of energy changed in the spot and flex dispatch used to update the energy table for after_flex 
            times """
            self.flexChangedEnergy = self.energyTable.loc[time + 1, 'before_flex'] - self.energyTable.loc[time, 'before_flex']
            self.spotChangedEnergy = self.energyTable.loc[time + 1, 'after_spot'] - self.energyTable.loc[time, 'after_spot']
            if not time+1 == self.spotTimePeriod:
                if dispatched:
                    if qty >= 0:
                        possible, _ = self.canCharge(chargePower=qty, chargeTime=self.spotTimeInterval,
                                                            status='after_flex', index=time)
                        if possible:
                            self.changeSOC(qty, self.spotTimeInterval, 'after_flex', time+1)
                        else:
                            self.changeSOC(0, self.spotTimeInterval, 'after_flex', time + 1)
                            self.penalizeTimes.append(time)
                    else:
                        possible, _ = self.canDischarge(dischargePower=qty, dischargeTime=self.spotTimeInterval,
                                                               status='after_flex', index=time)
                        if possible:
                            self.changeSOC(qty, self.spotTimeInterval, 'after_flex', time+1)
                        else:
                            self.changeSOC(0, self.spotTimeInterval, 'after_flex', time + 1)
                            self.penalizeTimes.append(time)
                else:
                    self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy
                    if not time+1 == self.spotTimePeriod:
                        self.energyTable.loc[time+1, 'after_flex'] = self.remainingEnergy

        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
        the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1)*self.dailySpotTime
        self.energyTable.loc[nextDayFirstTime, ['after_spot', 'before_flex', 'after_flex']] = \
            self.energyTable.loc[nextDayFirstTime, 'after_flex']
        self.remainingEnergy = self.energyTable.loc[nextDayFirstTime, 'after_flex']

        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

    def flexMarketReward(self, time, qty, price):
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty
        self.dailyRewardTable.loc[self.penalizeTimes, 'reward_flex'] += self.penaltyViolation
        totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
            = self.dailyRewardTable.loc[:, 'reward_flex']
        self.updateReward(totalReward)

    def makeBid(self, dailyBid, status):
        self.penalizeTimes = []
        for time, qty in zip(dailyBid['time'].values, dailyBid['qty_bid'].values):
            if status == 'before_flex':
                """amount of energy changed in the spot dispatch used to update the energy table for before_flex times"""
                self.spotChangedEnergy = self.energyTable.loc[time+1, 'after_spot'] - self.energyTable.loc[time, 'after_spot']

            if qty <= 0:
                # power output to grid - generator
                # constraintAmount is positive
                possible,_ = self.canDischarge(dischargePower=qty, dischargeTime=self.spotTimeInterval,
                                                               status=status, index=time)
                if possible:
                    self.changeSOC(qty, self.spotTimeInterval, status, time+1)
                else:
                    """If charging by a certain amount is not possible, change the bid to 0 and penalize the agent"""
                    dailyBid.loc[time, 'qty_bid'] = 0
                    self.changeSOC(0, self.spotTimeInterval, status, time + 1)
                    if status == 'after_spot':
                        self.penalizeTimes.append(time)

            elif qty > 0:
                # power intake from grid - consumer
                # constraintAmount is positive
                possible, _ = self.canCharge(chargePower=qty, chargeTime=self.spotTimeInterval,
                                                            status=status, index=time)
                if possible:
                    self.changeSOC(qty, self.spotTimeInterval, status, time+1)
                else:
                    """If charging by a certain amount is not possible, change the bid to 0 and penalize the agent"""
                    dailyBid.loc[time, 'qty_bid'] = 0
                    self.changeSOC(0, self.spotTimeInterval, status, time + 1)
                    if status == 'after_spot':
                        self.penalizeTimes.append(time)

            assert -self.maxPower <= dailyBid.loc[time, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'
