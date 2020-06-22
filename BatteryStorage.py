import numpy as np
import pandas as pd
from FlexAgent import FlexAgent


class BatStorage(FlexAgent):
    def __init__(self, id, location=[0, 0], maxPower=0, marginalCost=0,
                 maxCapacity=0, efficiency=1.0, SOC=1.0, minSOC=0.2):

        super().__init__(id=id, location=location, maxPower=maxPower, marginalCost=marginalCost)
        self.type = "Battery Storage"
        self.maxCapacity = maxCapacity  # capacity in MWh
        """
        How much energy is left in the battery after the spot market bids
        and after the spot market dispatch
        used to update the energyTable
        """
        self.remEnergyBeforeSpot = self.maxCapacity
        self.remEnergyAfterSpot = self.maxCapacity

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

        self.reset()

    def reset(self):
        super().reset()
        self.remEnergyBeforeSpot = self.maxCapacity
        self.remEnergyAfterSpot = self.maxCapacity
        self.SOC = 1.0

        # TODO remove the random initialization later
        self.spotBidMultiplier = np.random.uniform(-1.2, 1.2, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(-1.2, 1.2, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(1, 5, size=self.dailySpotTime)

        self.energyTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'before_spot': np.full(self.spotTimePeriod, self.remEnergyBeforeSpot,
                                                                     dtype=float),
                                              'dispatch_spot': np.full(self.spotTimePeriod, False),
                                              'after_spot': np.full(self.spotTimePeriod, self.remEnergyAfterSpot,
                                                                    dtype=float),
                                              'realised_spot': np.full(self.spotTimePeriod, True),
                                              'before_flex': np.full(self.flexTimePeriod, self.remEnergyBeforeSpot,
                                                                     dtype=float),
                                              'dispatch_flex': np.full(self.spotTimePeriod, False),
                                              'after_flex': np.full(self.flexTimePeriod, self.remEnergyBeforeSpot,
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
        if status == 'before_spot':
            if self.remEnergyBeforeSpot + energy > self.maxCapacity:
                return False, self.remEnergyBeforeSpot + energy - self.maxCapacity
            else:
                return True, None
        elif status == 'before_flex':
            spotDispatchedQty = self.energyTable.loc[index, 'after_spot'] - self.energyTable.loc[index+1, 'after_spot']
            if spotDispatchedQty + self.energyTable.loc[index, status] + energy > self.maxCapacity:
                return False, spotDispatchedQty + self.energyTable.loc[index, status] + energy - self.maxCapacity
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
        if status == 'before_spot':
            if self.remEnergyBeforeSpot + energy < self.minCapacity:
                return False, self.minCapacity - (self.remEnergyBeforeSpot + energy)
            else:
                return True, None
        elif status == 'before_flex':
            """checks if in the current hour spot dispatched qty + remaining energy after flex bidding 
            leads to constraints violation or not"""
            spotDispatchedQty = self.energyTable.loc[index, 'after_spot'] - self.energyTable.loc[index+1, 'after_spot']
            if spotDispatchedQty + self.energyTable.loc[index, status] + energy < self.minCapacity:
                return False, self.minCapacity - (spotDispatchedQty + self.energyTable.loc[index, status] + energy)
            else:
                return True, None

    def changeSOC(self, qty, time, status, index):
        energy = qty * time * self.efficiency
        if status == 'before_spot':
            self.remEnergyBeforeSpot += energy
            self.SOC = self.remEnergyBeforeSpot / self.maxCapacity
            if not index == self.spotTimePeriod:
                self.energyTable.loc[index, status] = self.remEnergyBeforeSpot
        elif status == 'before_flex':
            if not index == self.spotTimePeriod:
                self.energyTable.loc[index, status] = self.energyTable.loc[index-1, status] + energy

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

        self.remEnergyAfterSpot = None
        for time, energy in zip(self.dailySpotBid['time'].values,
                                self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes),
                                                     'before_spot'].values):
            if self.remEnergyAfterSpot is None:
                self.remEnergyAfterSpot = energy
                self.energyTable.loc[time, 'after_spot'] = self.remEnergyAfterSpot

            if time in spotDispatchedTimes:
                dispatchedEnergy = spotDispatchedQty.loc[time] * self.timeInterval
            else:
                dispatchedEnergy = 0.0
            """
            energyTable holds energy at the start of the specified time
            so, a dispatch in current time affects the energy at the next time
            """
            if not time+1 == self.spotTimePeriod:
                # assert self.minCapacity <= (self.remEnergyAfterSpot + dispatchedEnergy) <= self.maxCapacity, \
                #     "The constraints on max/min capacity of battery is violated " \
                #     "after spot dispatch {}".format(self.remEnergyAfterSpot + dispatchedEnergy)
                # # TODO How to deal with this situation?

                if not self.minCapacity <= (self.remEnergyAfterSpot + dispatchedEnergy) <= self.maxCapacity:
                    """
                    if constraints are violated after the spot clearance, the agent cannot dispatch
                    """
                    # TODO Penalize the agent here
                    self.energyTable.loc[time, 'realised_spot'] = False
                    dispatchedEnergy = 0.0

                self.energyTable.loc[time + 1, 'after_spot'] = self.remEnergyAfterSpot + dispatchedEnergy
            self.remEnergyAfterSpot += dispatchedEnergy

        realised = self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'realised_spot'].values
        self.spotMarketReward(spotDispatchedTimes.values, spotDispatchedQty.values, realised)
        # set the remaining energies
        self.remEnergyBeforeSpot = self.remEnergyAfterSpot

    def spotMarketReward(self, time, qty, realisedStatus):
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        # Here negative of qty is used for reward because generation is negative qty
        self.dailyRewardTable.loc[time, 'reward_spot'] = (self.dailySpotBid.loc[time, 'MCP'] - self.marginalCost) * -qty
        """
        For the unrealised dispatch, a negative reward is given wrt to MCP for maxPower qty"""
        self.dailyRewardTable.loc[~realisedStatus, 'reward_spot'] = self.dailySpotBid.loc[~realisedStatus, 'MCP'] \
                                                               * -np.abs(self.maxPower)

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
        self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes), 'dispatch_flex'] = \
            self.dailyFlexBid.loc[:, 'dispatched']
        for time, energy, dispatched in zip(self.dailyFlexBid['time'].values,
                                            self.energyTable.loc[self.energyTable['time'].isin(self.dailyTimes),
                                                                 'before_flex'].values,
                                            self.dailyFlexBid.loc[:, 'dispatched'].values):
            if not time+1 == self.spotTimePeriod:
                if dispatched:
                    self.energyTable.loc[time+1, 'after_flex'] = self.energyTable.loc[time+1, 'before_flex']
                else:
                    # No change in energy
                    self.energyTable.loc[time+1, 'after_flex'] = self.energyTable.loc[time, 'after_flex']

        self.flexMarketReward(flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice)

    def makeBid(self, dailyBid, status):
        for i, qty in zip(dailyBid['time'].values, dailyBid['qty_bid'].values):
            """
            If charging by a certain amount is not possible, checks the limit of charging amount possible
                and selects a random quantity within the limit
                Note: In that case, we are not trying to make charge cycle to discharge cycle
            The same case for discharge 
            """
            if qty <= 0:
                # power output to grid - generator
                # constraintAmount is positive
                possible, constraintAmount = self.canDischarge(dischargePower=qty, dischargeTime=self.timeInterval,
                                                               status=status, index=i)
                if possible:
                    self.changeSOC(qty, self.timeInterval, status, i+1)
                else:
                    # reduce discharge at least by constraintAmount
                    dailyBid.loc[i, 'qty_bid'] += np.random.uniform(constraintAmount, self.maxPower)
                    self.changeSOC(dailyBid.loc[i, 'qty_bid'], self.timeInterval, status, i+1)
                    # TODO check other possible options here

            elif qty > 0:
                # power intake from grid - consumer
                # constraintAmount is positive
                possible, constraintAmount = self.canCharge(chargePower=qty, chargeTime=self.timeInterval,
                                                            status=status, index=i)
                if possible:
                    self.changeSOC(qty, self.timeInterval, status, i+1)
                else:
                    # reduce charge at least by constraintAmount
                    dailyBid.loc[i, 'qty_bid'] -= np.random.uniform(constraintAmount, self.maxPower)
                    self.changeSOC(dailyBid.loc[i, 'qty_bid'], self.timeInterval, status, i+1)

            assert -self.maxPower <= dailyBid.loc[i, 'qty_bid'] <= self.maxPower, \
                'Qty bid cannot be more than maxPower'
