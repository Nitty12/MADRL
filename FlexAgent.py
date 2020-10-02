import numpy as np
import pandas as pd


class FlexAgent:
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, marginalCost = 30,
                 dailySpotTime = 24, needCounterTrade = False,
                 startDay=0, endDay=365):
        self.id = id
        self.node = ""
        self.type = "unknown"
        self.location = location  #
        self.minPower = minPower
        self.maxPower = maxPower
        self.marginalCost = marginalCost
        self.startDay = startDay
        self.endDay = endDay
        self.dailySpotTime = dailySpotTime
        self.spotTimePeriod = (self.endDay - self.startDay)*self.dailySpotTime
        self.day = startDay
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        self.totalTimes = np.arange(self.startDay * self.dailySpotTime, self.endDay * self.dailySpotTime)
        self.dailyFlexTime = 24
        self.flexTimePeriod = (self.endDay - self.startDay)*self.dailySpotTime
        self.spotTimeInterval = 1  # in hours
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

        self.rewardCount = 0.0
        # self.NN = AgentNeuralNet()
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
        self.highPriceLimit = 2
        self.penaltyViolation = -100
        self.spotState = None
        """current day MCP"""
        self.MCP = []

    def reset(self, *args, **kwargs):
        self.day = args[0]
        self.dailyTimes = np.arange(self.day * self.dailySpotTime, (self.day + 1) * self.dailySpotTime)
        self.totalTimes = np.arange(self.startDay * self.dailySpotTime, self.endDay * self.dailySpotTime)
        self.spotBid = pd.DataFrame(data={'time': self.totalTimes,
                                          'qty_bid': np.full(self.spotTimePeriod, self.maxPower, dtype=float),
                                          'dispatched': np.full(self.spotTimePeriod, False),
                                          'MCP': np.full(self.spotTimePeriod, 0.0, dtype=float)},
                                    index=self.totalTimes)
        self.dailySpotBid = self.spotBid.loc[self.spotBid['time'].isin(self.dailyTimes)]
        self.flexBid = pd.DataFrame(data={'time': self.totalTimes,
                                          'qty_bid': np.full(self.flexTimePeriod, self.maxPower, dtype=float),
                                          'price': np.full(self.flexTimePeriod, self.marginalCost, dtype=float),
                                          'flex_capacity': np.full(self.flexTimePeriod, self.maxPower, dtype=float),
                                          'dispatched': np.full(self.flexTimePeriod, False)},
                                    index=self.totalTimes)
        self.dailyFlexBid = self.flexBid.loc[self.flexBid['time'].isin(self.dailyTimes)]
        self.rewardTable = pd.DataFrame(data={'time': self.totalTimes,
                                              'reward_spot': np.full(self.spotTimePeriod, 0, dtype=float),
                                              'reward_flex': np.full(self.spotTimePeriod, 0, dtype=float)},
                                    index=self.totalTimes)
        self.dailyRewardTable = self.rewardTable.loc[self.rewardTable['time'].isin(self.dailyTimes)]
        self.spotBidMultiplier = np.random.uniform(self.lowSpotBidLimit, self.highSpotBidLimit, size=self.dailySpotTime)
        self.flexBidMultiplier = np.random.uniform(self.lowFlexBidLimit, self.highFlexBidLimit, size=self.dailySpotTime)
        self.flexBidPriceMultiplier = np.random.uniform(self.lowPriceLimit, self.highPriceLimit, size=self.dailySpotTime)
        self.rewardCount = 0.0
        self.spotState = True
        self.MCP = list(np.random.randint(0, 20, size=24))

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

    def setMCP(self, price):
        self.MCP = price

    def getActionLimits(self, alg):
        """returns the min and max limits for the action of the agent"""
        # TODO will the same space work for both states?
        """DSM, eV, PV and wind have different action space for flex bid
            eg, PV flex bid will be positive to decrease the spot dispatched (-ve) amounts"""
        if alg =='QMIX' or alg == 'IQL':
            stepSize = 20
            dataType = int
            """since Q net only supports actions from (0 - n), need to map the negative actions later"""
            minSpotBidArray = np.full(self.dailySpotTime, 0, dtype=dataType)
            maxSpotBidArray = np.full(self.dailySpotTime, stepSize * (self.highSpotBidLimit - self.lowSpotBidLimit),
                                      dtype=dataType)
            minFlexBidArray = np.full(self.dailySpotTime, 0, dtype=dataType)
            maxFlexBidArray = np.full(self.dailySpotTime, stepSize * (self.highFlexBidLimit - self.lowFlexBidLimit),
                                      dtype=dataType)
            minPriceArray = np.full(self.dailySpotTime, 0, dtype=dataType)
            maxPriceArray = np.full(self.dailySpotTime, stepSize * (self.highPriceLimit - self.lowPriceLimit),
                                    dtype=dataType)
        else:
            dataType = float
            minSpotBidArray = np.full(self.dailySpotTime, self.lowSpotBidLimit, dtype=dataType)
            maxSpotBidArray = np.full(self.dailySpotTime, self.highSpotBidLimit, dtype=dataType)
            minFlexBidArray = np.full(self.dailySpotTime, self.lowFlexBidLimit, dtype=dataType)
            maxFlexBidArray = np.full(self.dailySpotTime, self.highFlexBidLimit, dtype=dataType)
            minPriceArray = np.full(self.dailySpotTime, self.lowPriceLimit, dtype=dataType)
            maxPriceArray = np.full(self.dailySpotTime, self.highPriceLimit, dtype=dataType)

        return np.hstack((minSpotBidArray, minFlexBidArray, minPriceArray)), \
               np.hstack((maxSpotBidArray, maxFlexBidArray, maxPriceArray))

    def getObservation(self, startDay=0):
        """returns the current observation of the environment:
            observation is
                hourly market clearing prices of 't' Day ahead market,
                hourly dispatched power of the agent in 't-1' Day ahead market,
                hourly dispatched power of the agent in 't-1' Flex market,
                spot or flex state
        """
        # TODO add reqd flex times?, add weekend vs weekdays?
        if self.day == startDay:
            spotDispatch = np.full(self.dailySpotTime, 0)
            flexDispatch = np.full(self.dailySpotTime, 0)
        else:
            prevDayTimes = np.arange((self.day-1) * self.dailySpotTime, self.day * self.dailySpotTime)

            spotDispatch = np.full(self.dailySpotTime, 0)
            flexDispatch = np.full(self.dailySpotTime, 0)

            dispatchedMask = self.spotBid.loc[self.spotBid['time'].isin(prevDayTimes), 'dispatched'].values
            spotDispatch[dispatchedMask] = self.spotBid.loc[self.spotBid['time'].isin(prevDayTimes),
                                                            'qty_bid'][dispatchedMask].values

            dispatchedMask = self.flexBid.loc[self.flexBid['time'].isin(prevDayTimes), 'dispatched'].values
            flexDispatch[dispatchedMask] = self.flexBid.loc[self.flexBid['time'].isin(prevDayTimes),
                                                            'qty_bid'][dispatchedMask].values

        return self.MCP, spotDispatch, flexDispatch, int(self.spotState)

    def getReward(self):
        spotReward = self.dailyRewardTable.loc[:, 'reward_spot'].sum()
        flexReward = self.dailyRewardTable.loc[:, 'reward_flex'].sum()
        return spotReward, flexReward

