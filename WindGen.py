from REAgent import REAgent


class WG(REAgent):
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, voltageLevel= 0, marginalCost = 30,
                 feedInPremium = 64, genSeries=None, startDay=0, endDay=365):
        super().__init__(id, location=location, minPower = minPower, maxPower = maxPower, voltageLevel= voltageLevel,
                         marginalCost = marginalCost, feedInPremium = feedInPremium, genSeries=genSeries,
                         startDay=startDay, endDay=endDay)
        self.type = "Wind Generation"
