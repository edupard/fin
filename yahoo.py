from yahoo_finance import Share

yahoo = Share('SBRCY')

data = yahoo.get_historical('2014-04-25', '2014-04-29')

i = 0