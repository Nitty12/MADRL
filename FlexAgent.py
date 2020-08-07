from AgentNeuralNet import AgentNeuralNet
import numpy as np
import pandas as pd


class FlexAgent:
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, marginalCost = 0,
                 spotTimePeriod = 8760, dailySpotTime = 24, flexTimePeriod = 0, needCounterTrade = False,
                 discountFactor = 0.7, learningRate = 0.001):
        self.id = id
        self.type = "unknown"
        self.location = location  #
        self.minPower = minPower
        self.maxPower = maxPower
        self.marginalCost = marginalCost
        self.spotTimePeriod = spotTimePeriod
        self.dailySpotTime = dailySpotTime
        self.day = None
        self.dailyTimes = None
        self.dailyFlexTime = 24
        self.flexTimePeriod = 8760
        self.spotTimeInterval = 1  # in hours
        self.flexTimeInterval = 0.25  # in hours
        self.needCounterTrade = needCounterTrade  # Is counter trading responsibility of FlexAgent?
        self.spotBid = None
        self.dailySpotBid = None
        self.flexBid = None
        self.dailyFlexBid = None
        self.rewardTable = None
        self.dailyRewardTable = None
        '''
        spotBidMultiplier is used to adapt the qty_bid in spot market
        '''
        self.spotBidMultiplier = None
        '''
        flexQtyMultiplier is used to adapt the qty_bid in flex market
        flexBidPriceMultiplier is used to adapt the price of bid in flex market
        '''
        self.flexBidMultiplier = None
        self.flexBidPriceMultiplier = None

        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.rewardCount = 0.0
        self.NN = AgentNeuralNet()
        """during flex bid some agents like 
            DSM may need to bid -ve qty to reduce the qty already purchased in spot market 
            eV may need to bid -ve qty to (eg, to stop/reduce charging)
            and some agents like PV may need to bid +ve qty to reduce the generation dispatched in spot market
                Assumption: the time series data for PV and Wind contains -ve qty (generation)"""
        self.lowSpotBidLimit = 0
        self.highSpotBidLimit = 1
        self.lowFlexBidLimit = -1
        self.highFlexBidLimit = 0
        self.lowPriceLimit = 1
        self.highPriceLimit = 5
        self.penaltyViolation = -100
        self.spotState = None

    def reset(self, *args, **kwargs):
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        self.spotBid = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                          'qty_bid': np.full(self.spotTimePeriod, self.maxPower, dtype=float),
                                          'dispatched': np.full(self.spotTimePeriod, False),
                                          'MCP': np.full(self.spotTimePeriod, 0.0, dtype=float)})
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]
        self.flexBid = pd.DataFrame(data={'time': np.arange(self.flexTimePeriod),
                                          'qty_bid': np.full(self.flexTimePeriod, self.maxPower, dtype=float),
                                          'price': np.full(self.flexTimePeriod, self.marginalCost, dtype=float),
                                          'flex_capacity': np.full(self.flexTimePeriod, self.maxPower, dtype=float),
                                          'dispatched': np.full(self.flexTimePeriod, False)})
        self.dailyFlexBid = self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)]
        self.rewardTable = pd.DataFrame(data={'time': np.arange(self.spotTimePeriod),
                                              'reward_spot': np.full(self.spotTimePeriod, 0, dtype=float),
                                              'reward_flex': np.full(self.spotTimePeriod, 0, dtype=float)})
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        self.spotBidMultiplier = np.random.uniform(self.lowSpotBidLimit, self.highSpotBidLimit, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(self.lowFlexBidLimit, self.highFlexBidLimit, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(self.lowPriceLimit, self.highPriceLimit, size=self.dailySpotTime)
        self.rewardCount = 0.0
        self.spotState = True

    def printInfo(self):
        print("Agent ID: {}\nLocation: {}\nMaximum Power: {}"
              .format(self.id, self.location, self.maxPower))

    def getID(self):
        return self.id

    def makeSpotBid(self):
        self.spotBidMultiplier = self.getSpotBidMultiplier()
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes), 'qty_bid'] *= self.spotBidMultiplier
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]

    def boundSpotBidMultiplier(self, low, high):
        """
        if -1, agent sells the maxPower
        if 1, agent buys the maxPower
        """
        # TODO does this affect learning?
        self.spotBidMultiplier[self.spotBidMultiplier < low] = low
        self.spotBidMultiplier[self.spotBidMultiplier > high] = high

    def getSpotBidMultiplier(self):
        # TODO get the sbm from the RL policy and update
        return self.spotBidMultiplier

    def spotMarketEnd(self):
        self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes), ['dispatched', 'MCP']] = \
            self.dailySpotBid.loc[:, ['dispatched', 'MCP']]
        spotDispatchedTimes = self.dailySpotBid.query('dispatched == True').loc[:, 'time']
        spotDispatchedQty = self.dailySpotBid.loc[spotDispatchedTimes, 'qty_bid']
        if self.type not in ["PV Generation", "Wind Generation"]:
            return spotDispatchedTimes, spotDispatchedQty

    def makeFlexBid(self, reqdFlexTimes):
        if self.needCounterTrade:
            # reduce the flex capacity for the spot dispatched times k
            # TODO adjust for total time
            spotDispatchedTimes = self.dailySpotBid.query('dispatched == True').loc[:, 'time']
            spotDispatchedQty = self.dailySpotBid.loc[spotDispatchedTimes, 'qty_bid']
            self.flexBid.loc[self.flexBid['time'].isin(spotDispatchedTimes), ['flex_capacity']] = \
                self.maxPower - spotDispatchedQty

        self.flexBidMultiplier, self.flexBidPriceMultiplier = self.getFlexBidMultiplier()
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'qty_bid'] *= self.flexBidMultiplier
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'price'] *= self.flexBidPriceMultiplier

        self.dailyFlexBid = self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)]
        self.dailyFlexBid.loc[~self.dailyFlexBid['time'].isin(reqdFlexTimes), ['qty_bid', 'price']] = 0

        # if self.type in ["Demand Side Management", "E-vehicle VPP"]:
        #     """if the flexbid is not between (-spot dispatched qty) and (maxPower-spot dispatched qty),
        #     its not valid """
        #     for i, qty in zip(self.dailyFlexBid['time'].values, self.dailyFlexBid['qty_bid'].values):
        #         lowLimit = 0
        #         highLimit = self.maxPower
        #         if self.dailySpotBid.loc[i, 'dispatched']:
        #             lowLimit = -self.dailySpotBid.loc[i, 'qty_bid']
        #             highLimit = self.maxPower - self.dailySpotBid.loc[i, 'qty_bid']
        #
        #         if not lowLimit <= qty <= highLimit:
        #             # TODO bid is not valid, penalize the agent?
        #             self.dailyFlexBid.loc[i, 'qty_bid'] = 0
        #             self.dailyFlexBid.loc[i, 'price'] = 0
        #     self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)] = self.dailyFlexBid

    def boundFlexBidMultiplier(self, low, high, priceLimit=None):
        """
        flexBidMultiplier should be between low and high
        flexBidPriceMultiplier maximum limit
        """
        # TODO does this affect learning?
        if priceLimit is None:
            priceLimit = self.highPriceLimit
        self.flexBidMultiplier[self.flexBidMultiplier < low] = low
        self.flexBidMultiplier[self.flexBidMultiplier > high] = high
        # TODO what happens with PV and Wind?
        if self.type not in ["PV Generation", "Wind Generation"]:
            self.flexBidPriceMultiplier[self.flexBidPriceMultiplier > priceLimit] = priceLimit

    def getFlexBidMultiplier(self):
        # TODO get the fbm from the RL policy and update
        return self.flexBidMultiplier, self.flexBidPriceMultiplier

    def flexMarketEnd(self):
        self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes), 'dispatched'] = \
            self.dailyFlexBid.loc[:, 'dispatched']
        flexDispatchedTimes = self.dailyFlexBid.query('dispatched == True').loc[:, 'time']
        flexDispatchedQty = self.dailyFlexBid.loc[flexDispatchedTimes, 'qty_bid']
        flexDispatchedPrice = self.dailyFlexBid.loc[flexDispatchedTimes, 'price']
        return flexDispatchedTimes, flexDispatchedQty, flexDispatchedPrice

    def getSpotBid(self):
        return self.dailySpotBid

    def getFlexBid(self, reqdFlexTimes):
        return self.dailyFlexBid.loc[self.dailyFlexBid['time'].isin(reqdFlexTimes)]

    def spotMarketReward(self, *args, **kwargs):
        pass

    def flexMarketReward(self,  *args, **kwargs):
        pass

    def updateReward(self, reward):
        self.rewardCount += reward

    def getTotalReward(self):
        """cumulative reward for the agent during the episode"""
        # TODO should the reward be scaled between 0-1?
        return self.rewardCount

    def getActionLimits(self):
        """returns the min and max limits for the action of the agent"""
        # TODO will the same space work for both states?
        """DSM, eV, PV and wind have different action space for flex bid
            eg, PV flex bid will be positive to decrease the spot dispatched (-ve) amounts"""
        minSpotBidArray = np.full(self.dailySpotTime, self.lowSpotBidLimit, dtype=float)
        maxSpotBidArray = np.full(self.dailySpotTime, self.highSpotBidLimit, dtype=float)
        minFlexBidArray = np.full(self.dailySpotTime, self.lowFlexBidLimit, dtype=float)
        maxFlexBidArray = np.full(self.dailySpotTime, self.highFlexBidLimit, dtype=float)

        minPriceArray = np.full(self.dailySpotTime, self.lowPriceLimit, dtype=float)
        maxPriceArray = np.full(self.dailySpotTime, self.highPriceLimit, dtype=float)

        return np.hstack((minSpotBidArray, minFlexBidArray, minPriceArray)), \
               np.hstack((maxSpotBidArray, maxFlexBidArray, maxPriceArray))

    def getObservation(self):
        """returns the current observation of the environment:
            observation is
                hourly market clearing prices of 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Flex market,
                spot or flex state
        """
        # TODO add reqd flex times?, add weekend vs weekdays?
        if self.day == 0:
            MCP = np.random.randint(15, 30, size=24)
            spotDispatch = np.full(self.dailySpotTime, 0)
            flexDispatch = np.full(self.dailySpotTime, 0)
        else:
            prevDayTimes = np.arange((self.day-1) * self.dailySpotTime, self.day * self.dailySpotTime)
            MCP = self.spotBid.loc[self.spotBid['time'].isin(prevDayTimes), 'MCP'].values

            spotDispatch = np.full(self.dailySpotTime, 0)
            flexDispatch = np.full(self.dailySpotTime, 0)

            dispatchedMask = self.spotBid.loc[self.spotBid['time'].isin(prevDayTimes), 'dispatched'].values
            spotDispatch[dispatchedMask] = self.spotBid.loc[self.spotBid['time'].isin(prevDayTimes),
                                                            'qty_bid'][dispatchedMask].values

            dispatchedMask = self.flexBid.loc[self.flexBid['time'].isin(prevDayTimes), 'dispatched'].values
            flexDispatch[dispatchedMask] = self.flexBid.loc[self.flexBid['time'].isin(prevDayTimes),
                                                            'qty_bid'][dispatchedMask].values

        return MCP, spotDispatch, flexDispatch, int(self.spotState)

    def getReward(self):
        spotReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        flexReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        return spotReward, flexReward

