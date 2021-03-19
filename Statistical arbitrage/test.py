from strategy_main import statistical_arbitrage
from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np
from IPython.display import clear_output
from optibook.synchronous_client import Exchange
import time
import logging

logger = logging.getLogger('client')
logger.setLevel('INFO')



hist_stocks = pd.read_csv('cointegration/data.csv')
bot = statistical_arbitrage(hist_stocks)
bot.start_trading()
