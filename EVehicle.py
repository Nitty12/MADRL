import numpy as np
import pandas as pd
import os


from FlexAgent import FlexAgent


class EVehicle(FlexAgent):
    '''
        Represents an electric vehicles
        It can provide:
            +ve flexibility: STOP charging
            -ve flexibility: START charging
    '''

    def __init__(self, id, location=[0, 0], maxPower = 0.0036, marginalCost = 0,
                 maxCapacity = 0, efficiency = 1.0, absenceTimes=None):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "E-vehicle"
        self.maxCapacity = maxCapacity  # capacity in MWh
        self.minCapacity = 0.2*self.maxCapacity
        self.maxPower = maxPower
        """in absence times, 1 indicates EV in use and 0 indicates EV in charging station"""
        self.absenceTimes = absenceTimes
        self.dailyAbsenceTimes = None
        self.consumption = None
        self.remainingEnergy = None
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0
        self.efficiency = efficiency
        """in flex , can "sell" qty ie, to reduce the qty from the spot dispatched amount (eg, to stop/reduce charging)
                    can buy qty to increase the qty from the spot dispatched amount --> not considered now"""
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 0
        self.reset()

    def reset(self):
        super().reset()
        self.remainingEnergy = 0
        self.flexChangedEnergy = 0
        self.spotChangedEnergy = 0
        self.energyTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'after_spot': np.full(self.spotTimePeriod, self.remainingEnergy,
                                                                     dtype=float),
                                              'before_flex': np.full(self.flexTimePeriod, self.remainingEnergy,
                                                                    dtype=float),
                                              'after_flex': np.full(self.flexTimePeriod, self.remainingEnergy,
                                                                    dtype=float)})
        # self.importTimeseries()
        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(self.lowSpotBidLimit, self.highSpotBidLimit, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(self.lowFlexBidLimit, self.highFlexBidLimit, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(self.lowPriceLimit, self.highPriceLimit, size=self.dailySpotTime)


    def printInfo(self):
        super().printInfo()
        print("Type: {}\n".format(self.type))

    # def importTimeseries(self):
    #     # TODO initialize the timeseries
    #     self.absenceTimes = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod)})
    #     self.consumption = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
    #                                           'load': np.full(self.spotTimePeriod, self.maxPower,
    #                                                                  dtype=float)})

    def changeSOC(self, qty, timeInterval, status, index):
        energy = qty*timeInterval
        if status == 'after_spot':
            if not self.remainingEnergy + energy > self.maxCapacity:
                self.remainingEnergy += energy
                if not index == self.spotTimePeriod:
                    self.energyTable.loc[index, status] = self.remainingEnergy
        elif status == 'before_flex':
            """energy will be negative"""
            if not self.remainingEnergy + self.spotChangedEnergy + energy < self.minCapacity:
                self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy + energy
                if not index == self.spotTimePeriod:
                    self.energyTable.loc[index, status] = self.remainingEnergy

    def checkSOC(self):
        if self.remainingEnergy >= 0.8*self.maxCapacity:
            return True
        else:
            return False

    def makeSpotBid(self):
        self.boundSpotBidMultiplier(low=self.lowSpotBidLimit, high=self.highSpotBidLimit)
        super().makeSpotBid()

    def spotMarketEnd(self):
        spotDispatchedTimes, spotDispatchedQty = super().spotMarketEnd()
        self.dailyAbsenceTimes = self.absenceTimes.loc[self.day, :]
        """convert dailyAbsenceTimes to 24 period if in 96 period"""
        self.dailyAbsenceTimes = [time for i, time in enumerate(self.dailyAbsenceTimes) if i%4==0]
        penalizeTimes, startingPenalty = self.checkPenalties(spotDispatchedTimes, spotDispatchedQty, 'after_spot')
        """change remaining energy to that of the starting time for this day to use in flex market"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time']==self.dailyTimes[0], 'after_spot']
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values, penalizeTimes, startingPenalty)

    def spotMarketReward(self, time, qty, penalizeTimes, startingPenalty):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]

        """Price of electricity bought: Here negative of qty is used for reward because generation is negative qty"""
        self.dailyRewardTable.loc[time, 'reward_spot'] = self.dailySpotBid.loc[time, 'MCP'] * -qty[~self.dailyAbsenceTimes]
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
            self.spotChangedEnergy = self.energyTable.loc[time + 1, 'after_spot'] - self.energyTable.loc[
                time, 'after_spot']
            if self.dailyAbsenceTimes[i]:
                pass
            else:
                self.changeSOC(qty, self.spotTimeInterval, 'before_flex', time + 1)

    def flexMarketEnd(self):
        flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice = super().flexMarketEnd()
        """change remaining energy to that of the starting time for this day to update after_flex energy table"""
        self.remainingEnergy = self.energyTable.loc[self.energyTable['time']==self.dailyTimes[0], 'after_spot']
        penalizeTimes, startingPenalty = self.checkPenalties(flexDispatchedTimes, flexDispatchedQty, 'after_flex')
        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice, penalizeTimes, startingPenalty)

        """After flex market ends for a day, the final energy must be updated for all columns in energy table as it is 
                the final realised energy after the day"""
        nextDayFirstTime = (self.day + 1) * self.dailySpotTime
        self.energyTable.loc[nextDayFirstTime, ['before_spot', 'before_flex', 'after_flex']] = \
            self.energyTable.loc[nextDayFirstTime, 'after_flex']
        """change remaining energy to that of the starting time for this day"""
        self.remainingEnergy = self.energyTable.loc[nextDayFirstTime, 'after_flex']

    def flexMarketReward(self, time, qty, price, penalizeTimes, startingPenalty):
        """Price of electricity bought: Here negative of qty is used for reward because generation is negative qty"""
        self.dailyRewardTable.loc[time, 'reward_flex'] = price * -qty[~self.dailyAbsenceTimes]
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

        if status == 'after_spot':
            for time, qty, i in zip(DispatchedTimes.values, DispatchedQty.values, range(self.dailySpotTime)):
                """if its dispatched in absent times, dont store the energy but penalize the agent"""
                if self.dailyAbsenceTimes[i]:
                    """to check if the EV is starting"""
                    if not self.dailyAbsenceTimes[i-1] or i == 0:
                        """check for sufficient charge in EV"""
                        sufficient = self.checkSOC()
                        if not sufficient:
                            startingPenalty += self.penaltyViolation
                    """Ev not connected to charging station, penalize to not charge during this time"""
                    # TODO check if this or just making the bid as 0 works well
                    penalizeTimes.append(time)
                    # TODO take care of consumption during this time
                else:
                    self.changeSOC(qty, self.spotTimeInterval, status, time+1)

        elif status == 'after_flex':
            for time, qty, dispatched, i in zip(self.dailyFlexBid['time'].values,
                                             self.dailyFlexBid['qty_bid'].values,
                                             self.dailyFlexBid.loc[:, 'dispatched'].values,
                                             range(self.dailySpotTime)):
                """amount of energy changed in the spot and flex dispatch used to update the energy table for after_flex 
                times """
                self.flexChangedEnergy = self.energyTable.loc[time + 1, 'before_flex'] - self.energyTable.loc[
                    time, 'before_flex']
                self.spotChangedEnergy = self.energyTable.loc[time + 1, 'after_spot'] - self.energyTable.loc[
                    time, 'after_spot']

                if self.dailyAbsenceTimes[i]:
                    """to check if the EV is starting"""
                    if not self.dailyAbsenceTimes[i - 1] or i == 0:
                        """check for sufficient charge in EV"""
                        sufficient = self.checkSOC()
                        if not sufficient:
                            startingPenalty += self.penaltyViolation
                    """Ev not connected to charging station, penalize to not charge during this time"""
                    # TODO check if this or just making the bid as 0 works well
                    penalizeTimes.append(time)
                    # TODO take care of consumption during this time
                else:
                    if dispatched:
                        """negative qty of flex dispatch"""
                        if not self.remainingEnergy + self.flexChangedEnergy < self.minCapacity:
                            self.remainingEnergy = self.remainingEnergy + self.flexChangedEnergy
                            if not time == self.spotTimePeriod:
                                self.energyTable.loc[time+1, status] = self.remainingEnergy
                    else:
                        """positive qty of spot dispatch"""
                        if not self.remainingEnergy + self.spotChangedEnergy > self.maxCapacity:
                            self.remainingEnergy = self.remainingEnergy + self.spotChangedEnergy
                            if not time == self.spotTimePeriod:
                                self.energyTable.loc[time+1, status] = self.remainingEnergy

        return penalizeTimes, startingPenalty

