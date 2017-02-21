# Reinforcement learning for financial time series

Project goal is to adopt reinforcement learning for financial time series. Underlying data represent minute time bar serie.
You can find some dataset examples in /data folder.

# Environment

Inspired by [Bloomberg stock trading game](https://www.bloomberg.com/features/2015-stock-chart-trading-game/) environment adaptor was built

* It get raw dataset, preprocess data (merge several time bars, configurable)
* Then render chart similar to "trading game" using OpenGl and grab pixel data via framebuffer object. Exponential moving average used to move window center and bounds smoothly (? parameters really affect human performance - looks like some magic happen here).
* Environment accept 3 actions: LONG, FLAT, SHORT

# Remarks

* Fin time series is obviously a process with high enthropy => non-stationery processes, hard to derieve stable policy in comparison with deterministic environments
* It looks like it's easy for human to play on large time bars (data enthropy is lower). But if you would try to trade in physical world you would need a lot of time to run this strategy (strategy for right people :))... If your goal is short term strategy - use comparative analysis: i.e. look at several stocks and select subset with better opportunities. Great [HOWTO](http://cs229.stanford.edu/proj2013/TakeuchiLee-ApplyingDeepLearningToEnhanceMomentumTradingStrategiesInStocks.pdf) example.

# TODO

* Use Yahoo finance api to generate datasets
* Investigate if bloomberg game == look forward. Don't make any assumptions - test it! Possible options: ask them, read minified js, test same datasets via Yahoo finance api
* More discrete actions, continous action space, leverage and money management via neural networks!
* Use volume data, ohlc information
* Continous trading and ticks data

# Installation

* [GLFW](http://www.glfw.org/)

```
conda create --name rl_fin python=3.5
source activate rl_fin
pip install -r requirements.txt
```

# Windows issues:

* How to [install](https://github.com/rybskej/atari-py) atari
* You can install opencv 3.2.0 for python 3.5 using this wheel [link](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)