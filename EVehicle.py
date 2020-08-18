import numpy as np
import pandas as pd
import os
import ast

from FlexAgent import FlexAgent


class EVehicle(FlexAgent):
    '''
        Represents an electric vehicles
        It can provide:
            +ve flexibility: STOP charging
            -ve flexibility: START charging
    '''

    def __init__(self, id, location=[0, 0], maxPower = 0.0036, marginalCost = 0,
                 maxCapacity = 0, efficiency = 1.0, absenceTimes=None, consumption=None):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "E-vehicle"
        self.maxCapacity = maxCapacity  # capacity in MWh
        self.minCapacity = 0.2*self.maxCapacity
        self.maxPower = maxPower
        """in absence times, 1 indicates EV in use and 0 indicates EV in charging station"""
        self.absenceTimes = absenceTimes
        self.dailyAbsenceTimes = None
        self.consumption = consumption
        self.energyTable = None
        self.dailyConsumption = None
        self.remainingEnergy = None
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0
        self.efficiency = efficiency
        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount (eg, to stop/reduce charging)
                    can buy qty to increase the qty from the spot dispatched amount"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 1
        self.reset()

    def reset(self):
        super().reset()
        self.remainingEnergy = self.minCapacity
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0
        self.energyTable = pd.DataFrame(data={'time': np.arange(self.dailySpotTime+1),
                                              'after_spot': np.full(self.dailySpotTime+1, self.remainingEnergy,
                                                                     dtype=float),
                                              'before_flex': np.full(self.dailySpotTime+1, self.remainingEnergy,
                                                                    dtype=float),
                                              'after_flex': np.full(self.dailySpotTime+1, self.remainingEnergy,
                                                                    dtype=float)})

    def printInfo(self):
        super().printInfo()
        print("Type: {}\n".format(self.type))

    def changeSOC(self, qty, timeInterval, status, index):
        energy = qty*timeInterval
        if status == 'after_spot':
            if energy < 0:
                if not self.remainingEnergy + energy < self.minCapacity:
                    self.remainingEnergy += energy
            else:
                """if its greater than maxCapacity, cant store but has to pay for the electricity"""
                if not self.remainingEnergy + energy > self.maxCapacity:
                    self.remainingEnergy += energy
            if not index == self.spotTimePeriod:
                self.energyTable.loc[self.energyTable['time'] == index, status] = self.remainingEnergy

        elif status == 'before_flex':
            if energy < 0:
                if not self.remainingEnergy + self.spotChangedEnergy + energy < self.minCapacity:
                    self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy + energy
            else:
                if not self.remainingEnergy + self.spotChangedEnergy + energy > self.maxCapacity:
                    self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy + energy
            if not index == self.spotTimePeriod:
                self.energyTable.loc[self.energyTable['time'] == index, status] = self.remainingEnergy
        elif status == 'after_flex':
            if energy < 0:
                if not self.remainingEnergy + self.flexChangedEnergy < self.minCapacity:
                    self.remainingEnergy = self.remainingEnergy + self.flexChangedEnergy
            else:
                if not self.remainingEnergy + self.flexChangedEnergy > self.maxCapacity:
                    self.remainingEnergy = self.remainingEnergy + self.flexChangedEnergy
            if not index == self.spotTimePeriod:
                self.energyTable.loc[self.energyTable['time'] == index, status] = self.remainingEnergy

    def checkSOC(self):
        if self.remainingEnergy >= 0.8*self.maxCapacity:
            return True
        else:
            return False

    def makeSpotBid(self):
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()
        self.energyTable.loc[:, 'time'] = np.concatenate([self.dailyTimes, np.array([self.dailyTimes[-1] + 1])])

    def spotMarketEnd(self):
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.dailyAbsenceTimes = ast.literal_eval(self.absenceTimes.loc[self.day])
        self.dailyConsumption = ast.literal_eval(self.consumption.loc[self.day])
        self.convertToHourly()
        penalizeTimes, startingPenalty = self.checkPenalties(spotDispatchedTimes, spotDispatchedQty, 'after_spot')
        """change remaining energy to that of the starting time for this day to use in flex market"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time']==self.dailyTimes[0], 'after_spot'].values[0]
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values, penalizeTimes, startingPenalty)

    def spotMarketReward(self, time, qty, penalizeTimes, startingPenalty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        qty = [q if not self.dailyAbsenceTimes[i] else 0 for i, q in enumerate(qty)]
        """Price of electricity bought: Here negative of qty is used for reward because generation is negative qty"""
        self.dailyRewardTable.loc[time, 'reward_spot'] = self.dailySpotBid.loc[time, 'MCP'] * -np.array(qty)
        """Penalizing agent if not fully charged before departure"""
        self.dailyRewardTable.loc[time[0], 'reward_spot'] += startingPenalty
        """Penalize to not charge during absent times"""
        self.dailyRewardTable.loc[penalizeTimes, 'reward_spot'] += self.penaltyViolation

        totalReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_spot'] = \
            self.dailyRewardTable.loc[:, 'reward_spot']
        self.updateReward(totalReward)

    def makeFlexBid(self, reqdFlexTimes):
        self.boundFlexBidMultiplier(low=self.lowFlexBidLimit, high=self.highFlexBidLimit)
        """copy spot bids so that the flexbid multiplier is applied on this instead of maxPower"""
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'qty_bid'] = self.dailySpotBid.loc[:, 'qty_bid']
        super().makeFlexBid(reqdFlexTimes)
        for time, qty, i in zip(self.dailyFlexBid['time'].values,
                                self.dailyFlexBid['qty_bid'].values, range(self.dailySpotTime)):
            self.spotChangedEnergy = self.energyTable.loc[self.energyTable['time'] == time + 1, 'after_spot'].values[0] - \
                                     self.energyTable.loc[self.energyTable['time'] == time, 'after_spot'].values[0]
            if self.dailyAbsenceTimes[i]:
                self.changeSOC(0, self.spotTimeInterval, 'before_flex', time + 1)
            else:
                self.changeSOC(qty, self.spotTimeInterval, 'before_flex', time + 1)

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice = super().flexMarketEnd()
        """change remaining energy to that of the starting time for this day to update after_flex energy table"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time'] == self.dailyTimes[0], 'after_spot'].values[0]
        penalizeTimes, startingPenalty = self.checkPenalties(flexDispatchedTimes, flexDispatchedQty, 'after_flex')
        self.flexMarketReward(flexDispatchedTimes.values, flexDispatchedQty.values, flexDispatchedPrice.values,
                              penalizeTimes, startingPenalty)

        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
                the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1) * self.dailySpotTime
        self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, ['after_spot', 'before_flex']] = \
            self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, 'after_flex']
        """keep the energy at the end of last day as the starting energy"""
        self.energyTable.loc[0, ['after_spot', 'before_flex', 'after_flex']] = self.remainingEnergy
        """change remaining energy to that of the starting time for this day"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time'] == nextDayFirstTime, 'after_flex'].values[0]

    def flexMarketReward(self, time, qty, price, penalizeTimes, startingPenalty):
        if len(time) > 0:
            qty = [q if not self.dailyAbsenceTimes[i] else 0 for i, q in enumerate(qty)]
            """Price of electricity bought: Here negative of qty is used for reward because generation is negative qty"""
            self.dailyRewardTable.loc[time, 'reward_flex'] = price * -np.array(qty)
            """Penalizing agent if not fully charged before departure"""
            self.dailyRewardTable.loc[time[0], 'reward_flex'] += startingPenalty
            """Penalize to not offer flexibility during absent times"""
            self.dailyRewardTable.loc[penalizeTimes, 'reward_flex'] += self.penaltyViolation

            totalReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
            self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes), 'reward_flex'] \
                = self.dailyRewardTable.loc[:, 'reward_flex']
            self.updateReward(totalReward)

    def checkPenalties(self, DispatchedTimes, DispatchedQty, status):
        """penalizing to not charge at absent times"""
        penalizeTimes = []
        """penalizing for not having enough charge while starting"""
        startingPenalty = 0
        nStart = -1
        if status == 'after_spot':
            for time, qty, i in zip(DispatchedTimes.values, DispatchedQty.values, range(self.dailySpotTime)):
                """if its dispatched in absent times, dont store the energy but penalize the agent"""
                if self.dailyAbsenceTimes[i]:
                    nStart, startingPenalty, penalizeTimes = self.starting(nStart, startingPenalty, penalizeTimes,
                                                                           qty, i, status, time)
                else:
                    self.changeSOC(qty, self.spotTimeInterval, status, time+1)

        elif status == 'after_flex':
            for time, qty, dispatched, i in zip(self.dailyFlexBid['time'].values,
                                             self.dailyFlexBid['qty_bid'].values,
                                             self.dailyFlexBid.loc[:, 'dispatched'].values,
                                             range(self.dailySpotTime)):
                """amount of energy changed in the spot and flex dispatch used to update the energy table for after_flex 
                times """
                self.flexChangedEnergy = self.energyTable.loc[self.energyTable['time'] == time + 1, 'before_flex'].values[0] \
                                         - self.energyTable.loc[self.energyTable['time'] == time, 'before_flex'].values[0]
                self.spotChangedEnergy = self.energyTable.loc[self.energyTable['time'] == time + 1, 'after_spot'].values[0] \
                                         - self.energyTable.loc[self.energyTable['time'] == time, 'after_spot'].values[0]

                if self.dailyAbsenceTimes[i]:
                    nStart, startingPenalty, penalizeTimes = self.starting(nStart, startingPenalty, penalizeTimes,
                                                                           qty, i, status, time)
                else:
                    if dispatched:
                        """negative qty of flex dispatch"""
                        if not self.remainingEnergy + self.flexChangedEnergy < self.minCapacity:
                            self.remainingEnergy = self.remainingEnergy + self.flexChangedEnergy
                        if not time == self.spotTimePeriod:
                            self.energyTable.loc[self.energyTable['time'] == time+1, status] = self.remainingEnergy
                    else:
                        """positive qty of spot dispatch"""
                        if not self.remainingEnergy + self.spotChangedEnergy > self.maxCapacity:
                            self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy
                        if not time == self.spotTimePeriod:
                            self.energyTable.loc[self.energyTable['time'] == time+1, status] = self.remainingEnergy

        return penalizeTimes, startingPenalty

    def starting(self, nStart, startingPenalty, penalizeTimes, qty, i, status, time):
        """to check if the EV is starting"""
        if not self.dailyAbsenceTimes[i - 1] or i == 0:
            """EV is starting now"""
            nStart += 1
            """check for sufficient charge in EV"""
            sufficient = self.checkSOC()
            if not sufficient:
                startingPenalty += self.penaltyViolation
            """the consumption during entire trip"""
            if nStart < len(self.dailyConsumption):
                """data has sometimes less consumption values than the number of starts"""
                self.changeSOC(-self.dailyConsumption[nStart], self.spotTimeInterval, status, time + 1)
            else:
                self.changeSOC(0, self.spotTimeInterval, status, time + 1)
        else:
            self.changeSOC(0, self.spotTimeInterval, status, time + 1)
        """Ev not connected to charging station, penalize to not charge during this time"""
        if not qty == 0:
            # TODO check if this or just making the bid as 0 works well
            penalizeTimes.append(time)
        return nStart, startingPenalty, penalizeTimes

    def convertToHourly(self):
        """convert dailyAbsenceTimes to 24 period if in 96 period"""
        # self.dailyAbsenceTimes = [time for i, time in enumerate(self.dailyAbsenceTimes) if i%4==0]
        dailyAbsenceTimes =[]
        absent = 0
        for i, time in enumerate(self.dailyAbsenceTimes):
            if (i+1) % 4 == 0:
                if absent > 1:
                    dailyAbsenceTimes.append(1)
                else:
                    dailyAbsenceTimes.append(0)
                absent = 0
            else:
                absent += time
        self.dailyAbsenceTimes = dailyAbsenceTimes
