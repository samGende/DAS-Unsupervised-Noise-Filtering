import datetime as dt 
startTimes = [dt.datetime(2016,9,4,23,55,54,932000) ]
secondsPerFile = 60
secondsPerWindowWidth =300
#how much to shift the start time when performing cwt
secondsPerWindowOffset =150
xCorrMaxTimeLagSeconds = 3.0
nFiless = [720] 
outfilePath = '../scratch/'
outfileListFile = 'inter_results/outfileTraining.txt'
srcChList = 'sourceList.txt'
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0