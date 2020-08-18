import numpy as np


class FlexMarket:
    def __init__(self):
        self.participants = []
        self.bids = {}
        self.reqdFlexTimes = None
        self.dailyFlexTime = 24
        self.day = 0
        self.dailyTimes = None
        self.reset()

    def reset(self):
        self.day = 0
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)

    def addParticipants(self, participants):
        # participants is a list of FlexAgents
        self.participants.extend(participants)

    def collectBids(self, reqdFlexTimes):
        self.reqdFlexTimes = reqdFlexTimes
        for participant in self.participants:
            participant.makeFlexBid(self.reqdFlexTimes)
            assert len(participant.getFlexBid(self.reqdFlexTimes).index) == len(self.reqdFlexTimes),\
                'Length of flex market and bid not equal'
            self.bids[participant.id] = participant.getFlexBid(self.reqdFlexTimes)

    def sendDispatch(self, flexDispatchStatus):
        for participant in self.participants:
            if flexDispatchStatus is not None:
                self.bids[participant.id].loc[:, 'dispatched'] = flexDispatchStatus.loc[:, participant.id].values
                # setting explicitly
                participant.dailyFlexBid.loc[participant.dailyFlexBid['time'].isin(self.reqdFlexTimes), 'dispatched'] = \
                    flexDispatchStatus.loc[:, participant.id].values
            participant.flexMarketEnd()

    def endDay(self):
        # update day and dailyTimes after a day even if flex is not needed
        self.day += 1
        self.dailyTimes = np.arange(self.day * self.dailyFlexTime, (self.day + 1) * self.dailyFlexTime)
