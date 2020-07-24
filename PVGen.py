from REAgent import REAgent


class PVG(REAgent):
    def __init__(self, id, location=None, minPower = 0, maxPower = 0, voltageLevel= 0, marginalCost = 0,
                 feedInPremium = 64, genSeries=None):
        super().__init__(id, location=location, minPower = minPower, maxPower = maxPower, voltageLevel= voltageLevel,
                         marginalCost = marginalCost, feedInPremium = feedInPremium, genSeries=genSeries)
        self.type = "PV Generation"
