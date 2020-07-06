import numpy as np


class FlexMarket:
    def __init__(self):
        self.participants = []
        self.bids = {}
        self.reqdFlexTimes = None
        self.dailyFlexTime = 24
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)

    def reset(self):
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)

    def getCongestionStatus(self, grid):
        status = grid.checkCongestion(self.dailyTimes)
        # dailyGridStatus = grid.status.loc[grid.status['time'].isin(self.dailyTimes)]
        # status = np.any(dailyGridStatus.loc[:, 'congestion'])
        return status

    def addParticipants(self, participants):
        # participants is a list of FlexAgents
        self.participants.extend(participants)

    def collectBids(self, grid):
        # time periods in which flexibility is needed
        self.reqdFlexTimes = grid.getStatus().loc[grid.getStatus()['time']
            .isin(self.dailyTimes)].query('congestion == True')['time']
        for participant in self.participants:
            participant.makeFlexBid(self.reqdFlexTimes)
            assert len(participant.getFlexBid(self.reqdFlexTimes).index) == len(self.reqdFlexTimes),\
                'Length of flex market and bid not equal'
            self.bids[participant.getID()] = participant.getFlexBid(self.reqdFlexTimes)

    def sendDispatch(self):
        for participant in self.participants:
            # just a random boolean assignment for testing purpose
            flexDispatchStatus = np.random.choice([True, False], size=len(self.reqdFlexTimes.values))
            self.bids[participant.getID()].loc[:, 'dispatched'] = flexDispatchStatus
            # setting explicitly
            participant.dailyFlexBid.loc[participant.dailyFlexBid['time'].isin(self.reqdFlexTimes), 'dispatched'] = \
                flexDispatchStatus

            participant.flexMarketEnd()

    def endDay(self):
        # update day and dailyTimes after a day even if flex is not needed
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)
