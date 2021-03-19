from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np
import random
import sys
from IPython.display import clear_output
from optibook.synchronous_client import Exchange
import time 
import logging

logger = logging.getLogger('client')
logger.setLevel('INFO')




class statistical_arbitrage:
    
    def __init__(self , historical_data):
        
        self.e = Exchange()
        logging.info(self.e.connect())
        logging.info("Setup was successful.")
        self.get_out_of_positions()
        
        self.data = historical_data
        self.data.drop("Unnamed: 0", axis = 1 , inplace = True)

        self.pairs = [['TOTAL', 'UNILEVER'], ['UNILEVER', 'TOTAL'], ['SAP', 'ASML'], ['ASML', 'SAP'], ['LVMH', 'ALLIANZ'], ['ALLIANZ', 'LVMH']] # Obtained from 'find_cointegrated_pairs.py'


        self.cs = {}
        self.gammas = {}
        self.stds = {}
        self.alphas = {}
    
        for pair in self.pairs:
            
            y = np.log(self.data[pair[0]])
            x = np.log(self.data[pair[1]])
            c, gamma, alpha , z = estimate_long_run_short_run_relationships(y, x)
            self.gammas[tuple(pair)] = gamma
            self.alphas[tuple(pair)] = alpha
            self.stds[tuple(pair)] = np.std(z)
            self.cs[tuple(pair)] = c
            
            
        self.df = pd.DataFrame(columns = ['Pair', 'st_1', 'st_1_vol', 'st_1_price', 'st_2', 'st_2_vol', 'st_2_price'])
        self.hist_prof = []
        self.thresholds = [1.1]
        self.strat = [0] * len(self.pairs)
        self.profits = pd.DataFrame( columns = ['Pair', 'Instrument', 'Volume', 'VWAP', 'Price', 'Profit'] )
        
        
        for pair in self.pairs:
            self.df.loc[ self.pairs.index(pair)] = ([pair, pair[0], 0, 0, pair[1], 0, 0])



    def get_out_of_positions(self):
        # Get out of all positions you are currently holding, regardless of the loss involved. That means selling whatever
        # you are long, and buying-back whatever you are short. Be sure you know what you are doing when you use this logic.
        print(self.e.get_positions())
        
        while sum(self.e.get_positions().values()) != 0:
            
            for s, p in self.e.get_positions().items():
                if p > 0:
                    self.e.insert_order(s, price=1, volume=p, side='ask', order_type='ioc')
                elif p < 0:
                    self.e.insert_order(s, price=100_000, volume=-p, side='bid', order_type='ioc')  
            time.sleep(0.5)
            
        print(self.e.get_positions())
        
        
        
    def get_data(self, instrument):
    
        max_bid_tick = None
        min_ask_tick = None
        book = self.e.get_last_price_book(instrument)
        ask_prices = []
        bid_prices = []
    
        for tick in book.asks:
            ask_prices.append(tick.price)
            
        if ask_prices:
            min_ask_loc = np.argmin(ask_prices)
            min_ask_tick = book.asks[min_ask_loc]
            
        for tick in book.bids:
            bid_prices.append(tick.price)
            
        if bid_prices:
            max_bid_loc = np.argmax(bid_prices)
            max_bid_tick = book.bids[max_bid_loc]
            
        if min_ask_tick and max_bid_tick:
            return min_ask_tick, max_bid_tick
            
        else:
            return False, False
        
    
    
    def update_parameters(self, frequency):
        
        """
        This function updates the historical data with real-time prices and re-calculates the values for the parameters 'c', 'gamma', 'alpha' and 'z_std' at a fixed frequency.
        If the original pairs remain cointegrated, we shouldn't expect the values of these parameters to change a lot.
        
        Parameters
        ----------
        frequency : int
            Number of hours required to pass for parameters to be updated
        
        """
        
        i = len(self.data) + 1
        get_new_prices = [pair[0] for pair in self.pairs] 
        start = time.time()
        
        while True:
            
            for stock in get_new_prices:
                tick1 , tick2 = self.get_data(stock)
                if tick1 and tick2:
                    self.data.loc[i,stock] = (tick1.price + tick2.price) / 2 # Use midprice as estimate
                else:
                    self.data.loc[i,stock] = self.data.loc[i - 1 ,stock] # If price is not available, asign previous price to avoid NaNs
                    
            i += 1
            time.sleep(1)
            
            if ( (time.time() - start) / 60**2 ) > frequency: 
    
                for pair in self.pairs:
                    
                    y = np.log(self.data[pair[0]])
                    x = np.log(self.data[pair[1]])
                    c, gamma, alpha , z = estimate_long_run_short_run_relationships(y, x)
                    self.gammas[tuple(pair)] = gamma
                    self.alphas[tuple(pair)] = alpha
                    self.stds[tuple(pair)] = np.std(z)
                    self.cs[tuple(pair)] = c
                
                start = time.time() # reset the timer and continue
            
            
    def trade(self, thresholds):
        
        limit = 500
        
        
        for pair in self.pairs:
            
            a = random.choice([0, 1])
            pos = self.e.get_positions()
            st_1 = pair[0]
            st_2 = pair[1]
            gamma = self.gammas[tuple(pair)]
            c = self.cs[tuple(pair)]
            std = self.stds[tuple(pair)]
    
            upper_threshold = thresholds[-1]
            lower_threshold = 0.2
    
            ask_st_1, bid_st_1 = self.get_data(st_1)
            ask_st_2, bid_st_2 = self.get_data(st_2)
        
    
            if ask_st_1 and bid_st_1 and ask_st_2 and bid_st_2:
                
                st_1_mid = (ask_st_1.price + bid_st_1.price) / 2
                st_2_mid = (ask_st_2.price + bid_st_2.price) / 2
                z_val = np.log(st_1_mid) - c - (gamma * np.log(st_2_mid))
                hedge_ratio = gamma * (bid_st_1.price / ask_st_2.price) # how much of st_2 for every st_1
        
                if z_val > 0: # Use the positive spread to profit
            
                    st_1_id = None
                    st_2_id = None
            
                    if z_val > upper_threshold * std: # If spread is high enough, trade
        
                        vol_st_2 = np.floor(min((bid_st_1.volume * hedge_ratio), ask_st_2.volume, limit - pos[st_2], (limit + pos[st_1]) * hedge_ratio)) # how much of st_2 to buy
                        vol_st_1 = np.floor(vol_st_2 / hedge_ratio)
                        
                        
                        if vol_st_1 != 0 and vol_st_2 != 0:
                            
                            if (a == 0 and self.strat[ self.pairs.index(pair)] == 0) or self.strat[ self.pairs.index(pair)] == 1:
                                
                                st_1_id = self.e.insert_order(st_1, price = bid_st_1.price , volume = int(vol_st_1), side='ask', order_type='ioc')
                            
                            if st_1_id:
                                
                                tr_1_vol =  self.e.get_positions()[st_1] - pos[st_1]
                                vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price'] # volume weighted averge price
                                vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                
                                if tr_1_vol != 0:
                                    
                                    self.df.at[ self.pairs.index(pair), 'st_1_price'] = ((tr_1_vol * bid_st_1.price) + (vwap * vol)) / (tr_1_vol + vol)
                                    
                                    if - tr_1_vol != vol_st_1:
                                        
                                        vol_st_2 = np.floor(- tr_1_vol * hedge_ratio)
                                        
                                    if vol_st_2 != 0:
                                        
                                        st_2_id = self.e.insert_order(st_2, price=ask_st_2.price, volume=int(vol_st_2), side='bid', order_type='ioc')
                                        
                                        if st_2_id:
                                            
                                            tr_2_vol =  self.e.get_positions()[st_2] - pos[st_2]
                                            vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                            vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                            self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                            
                                            if tr_2_vol != 0:
                                                
                                                self.df.at[ self.pairs.index(pair), 'st_2_price'] = ((tr_2_vol * ask_st_2.price) + (vwap * vol)) / (tr_2_vol + vol)
                        
                        else:
                            
                            st_2_id = self.e.insert_order(st_2, price = ask_st_2.price, volume = int(vol_st_2), side='bid', order_type='ioc')
                            
                            if st_2_id:
                                
                                tr_2_vol =  self.e.get_positions()[st_2] - pos[st_2]
                                vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                
                                if tr_2_vol != 0:
                                    
                                    self.df.at[ self.pairs.index(pair), 'st_2_price'] = ((tr_2_vol * ask_st_2.price) + (vwap * vol)) / (tr_2_vol + vol)
                                    
                                    if tr_2_vol != vol_st_2:
                                        
                                        vol_st_1 = np.floor(tr_2_vol / hedge_ratio)
                                        
                                    if vol_st_1 != 0:
                                        
                                        st_1_id = self.e.insert_order(st_1, price = bid_st_1.price, volume = int(vol_st_1), side='ask', order_type='ioc')
                                        
                                        if st_1_id:
                                            
                                            tr_1_vol =  self.e.get_positions()[st_1] - pos[st_1]
                                            vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                            vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                            self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                            
                                            if tr_1_vol != 0:
                                                
                                                self.df.at[ self.pairs.index(pair), 'st_1_price'] = ((tr_1_vol * bid_st_1.price) + (vwap * vol)) / (tr_1_vol + vol)
                                    
                        
                if z_val < lower_threshold * std:
                    
                    if not self.df.empty:
                        
                        current_pos_1 = self.df.iloc[ self.pairs.index(pair)]['st_1_vol']
                        current_pos_2 = self.df.iloc[ self.pairs.index(pair)]['st_2_vol']
                        
                        if current_pos_1 != 0 or current_pos_2 != 0:
                            
                            vol_st_2 = np.floor(min(current_pos_2, ask_st_1.volume * hedge_ratio, bid_st_2.volume, (limit - pos[st_1]) * hedge_ratio, limit + pos[st_2]))
                            vol_st_1 = np.floor(min(vol_st_2 / hedge_ratio, - current_pos_1))
                            exp_prof = (vol_st_1 * (self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price)) + (vol_st_2 * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                            
                            if vol_st_1 != 0 and vol_st_2 != 0 and exp_prof > 0:
                                
                                if a == 0:
                                    
                                    st_1_id = self.e.insert_order(st_1, price=ask_st_1.price, volume=int(vol_st_1), side='bid', order_type='ioc')
                                    
                                    if st_1_id:
                                        
                                        tr_1_vol = self.e.get_positions()[st_1] - pos[st_1]
                                        vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                        vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                        prof = (tr_1_vol * ( self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price))
                                        self.profits.loc[st_1_id] = ( [pair, st_1, tr_1_vol, vwap, ask_st_1.price, prof] )
                                        self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                        
                                        if tr_1_vol+vol == 0:
                                            
                                            self.df.at[ self.pairs.index(pair), 'st_1_price'] = 0
                                            
                                        if tr_1_vol != vol_st_1: 
                                            
                                            vol_st_2 = np.floor(min(tr_1_vol * hedge_ratio, current_pos_2))
                                            
                                        if vol_st_2 != 0:
                                            
                                            st_2_id = self.e.insert_order(st_2, price = bid_st_2.price, volume = int(vol_st_2), side='ask', order_type='ioc')
                                            
                                            if st_2_id:
                                                
                                                tr_2_vol = self.e.get_positions()[st_2] - pos[st_2]
                                                vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                                vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                                prof = (- tr_2_vol * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                                                self.profits.loc[st_2_id] = ( [pair, st_2, tr_2_vol, vwap, bid_st_2.price, prof] )
                                                self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                                
                                                if tr_2_vol+vol == 0:
                                                    
                                                    self.df.at[ self.pairs.index(pair), 'st_2_price'] = 0
                                                    
                                else:
                                    
                                    st_2_id = self.e.insert_order(st_2, price = bid_st_2.price, volume = int(vol_st_2), side='ask', order_type='ioc')
                                    
                                    if st_2_id:
                                        
                                        tr_2_vol = self.e.get_positions()[st_2] - pos[st_2]
                                        vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                        vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                        prof = (- tr_2_vol * ( bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price'] ) )
                                        self.profits.loc[st_2_id] = ( [pair, st_2, tr_2_vol, vwap, bid_st_2.price, prof] )
                                        self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                        
                                        if tr_2_vol + vol == 0:
                                            
                                            self.df.at[ self.pairs.index(pair), 'st_2_price'] = 0
                                            
                                        if - tr_2_vol != vol_st_2: 
                                            
                                            vol_st_1 = np.floor(min(- tr_2_vol / hedge_ratio, - current_pos_1))
                                            
                                        if vol_st_1 != 0:
                                            
                                            st_1_id = self.e.insert_order(st_1, price = ask_st_1.price, volume = int(vol_st_1), side='bid', order_type='ioc')
                                            
                                            if st_1_id:
                                                
                                                tr_1_vol = self.e.get_positions()[st_1] - pos[st_1]
                                                vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                                vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                                prof = (tr_1_vol * ( self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price))
                                                self.profits.loc[st_1_id] = ( [pair, st_1, tr_1_vol, vwap, ask_st_1.price, prof] )
                                                self.df.at[self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                                
                                                if tr_1_vol+vol == 0:
                                                    
                                                    self.df.at[ self.pairs.index(pair), 'st_1_price'] = 0    
                                                    
                        
                if  z_val < 0:
                
                
                    if not self.df.empty:
                        
                        current_pos_1 = self.df.iloc[ self.pairs.index(pair)]['st_1_vol']
                        current_pos_2 = self.df.iloc[self.pairs.index(pair)]['st_2_vol']
                        
                        if current_pos_1 != 0 or current_pos_2 != 0:
                            
                            vol_st_2 = np.floor(min(current_pos_2, ask_st_1.volume * hedge_ratio, bid_st_2.volume, (limit - pos[st_1]) * hedge_ratio, limit + pos[st_2]))
                            vol_st_1 = np.floor(min(vol_st_2 / hedge_ratio, - current_pos_1))
                            exp_prof = (vol_st_1 * (self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price)) + (vol_st_2 * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                            
                            if vol_st_1 != 0 and vol_st_2 != 0 and exp_prof > 0:
                                
                                if a == 0:
                                    
                                    st_1_id = self.e.insert_order(st_1, price = ask_st_1.price, volume = int(vol_st_1), side='bid', order_type='ioc')
                                    
                                    if st_1_id:
                                        
                                        tr_1_vol = self.e.get_positions()[st_1] - pos[st_1]
                                        vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                        vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                        prof = (tr_1_vol * (self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price))
                                        self.profits.loc[st_1_id] = ( [pair, st_1, tr_1_vol, vwap, ask_st_1.price, prof] )
                                        self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                        
                                        if tr_1_vol+vol == 0:
                                            
                                            self.df.at[ self.pairs.index(pair), 'st_1_price'] = 0
                                            
                                        if tr_1_vol != vol_st_1: 
                                            
                                            vol_st_2 = np.floor(min(tr_1_vol * hedge_ratio, current_pos_2))
                                            
                                        if vol_st_2 != 0:
                                            
                                            st_2_id = self.e.insert_order(st_2, price = bid_st_2.price, volume = int(vol_st_2), side='ask', order_type='ioc')
                                            
                                            if st_2_id:
                                                
                                                tr_2_vol = self.e.get_positions()[st_2] - pos[st_2]
                                                vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                                vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                                prof = (- tr_2_vol * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                                                self.profits.loc[st_2_id] = ( [pair, st_2, tr_2_vol, vwap, bid_st_2.price, prof] )
                                                self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                                
                                                if tr_2_vol+vol == 0:
                                                    
                                                    self.df.at[ self.pairs.index(pair), 'st_2_price'] = 0
                                                    
                                else:
                                    
                                    st_2_id = self.e.insert_order(st_2, price = bid_st_2.price, volume = int(vol_st_2), side='ask', order_type='ioc')
                                    
                                    if st_2_id:
                                        
                                        tr_2_vol = self.e.get_positions()[st_2] - pos[st_2]
                                        vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                        vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                        prof = (- tr_2_vol * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                                        self.profits.loc[st_2_id] = ( [pair, st_2, tr_2_vol, vwap, bid_st_2.price, prof] )
                                        self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                        
                                        if tr_2_vol + vol == 0:
                                            
                                            self.df.at[ self.pairs.index(pair), 'st_2_price'] = 0
                                            
                                        if - tr_2_vol != vol_st_2: 
                                            
                                            vol_st_1 = np.floor(min(- tr_2_vol / hedge_ratio, - current_pos_1))
                                            
                                        if vol_st_1 != 0:
                                            
                                            st_1_id = self.e.insert_order(st_1, price = ask_st_1.price, volume = int(vol_st_1), side='bid', order_type='ioc')
                                            
                                            if st_1_id:
                                                
                                                tr_1_vol = self.e.get_positions()[st_1] - pos[st_1]
                                                vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                                vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                                prof = (tr_1_vol * ( self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price))
                                                self.profits.loc[st_1_id] = ( [pair, st_1, tr_1_vol, vwap, ask_st_1.price, prof])
                                                self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                                
                                                if tr_1_vol + vol == 0:
                                                    
                                                    self.df.at[ self.pairs.index(pair), 'st_1_price'] = 0        
                        
                        
                        
    def hedge_ratio_diff(self):
        
        hedge = pd.DataFrame(columns = ['Pair', 'Actual Hedge', 'Should be', 'Ratio'])
        
        for pair in self.pairs:
            
            pos = self.e.get_positions()
            limit = 500
            
            st_1 = pair[0]
            st_2 = pair[1]
            
            ask_st_1, bid_st_1 = self.get_data(st_1)
            ask_st_2, bid_st_2 = self.get_data(st_2)
            
            if ask_st_1 and bid_st_1 and ask_st_2 and bid_st_2:
                
                gamma = self.gammas[tuple(pair)]
                hedge_ratio = gamma * (bid_st_1.price / ask_st_2.price)
                st_1_tot = self.df.iloc[ self.pairs.index(pair)]['st_1_vol']
                st_2_tot = self.df.iloc[ self.pairs.index(pair)]['st_2_vol']
                
                if st_1_tot != 0 or st_2_tot != 0:
                    
                    if st_1_tot != 0:
                        actual_hedge = - (st_2_tot/st_1_tot)
                        
                    else:
                        actual_hedge = np.nan
                    
                    if ( actual_hedge / hedge_ratio > 10/9 ) or st_1_tot == 0:
                        
                        sell_st_2 = st_2_tot - ( - st_1_tot * hedge_ratio)
                        
                        if bid_st_2.price > self.df.iloc[ self.pairs.index(pair)]['st_2_price']:
                            
                            vol_st_2 = np.floor(min(sell_st_2, limit + pos[st_2]))
                            
                            if vol_st_2 != 0:
                                
                                st_2_id = self.e.insert_order(st_2, price = bid_st_2.price, volume = int(vol_st_2), side='ask', order_type='ioc')
                                
                                if st_2_id:
                                    
                                    tr_2_vol = self.e.get_positions()[st_2] - pos[st_2]
                                    vwap = self.df.loc[ self.pairs.index(pair)]['st_2_price']
                                    vol = self.df.loc[ self.pairs.index(pair)]['st_2_vol']
                                    prof = (- tr_2_vol * (bid_st_2.price - self.df.iloc[ self.pairs.index(pair)]['st_2_price']))
                                    self.profits.loc[st_2_id] = ( [pair, st_2, tr_2_vol, vwap, bid_st_2.price, prof] )
                                    self.df.at[ self.pairs.index(pair), 'st_2_vol'] += tr_2_vol
                                    
                                    if tr_2_vol+vol == 0:
                                        
                                        self.df.at[ self.pairs.index(pair), 'st_2_price'] = 0
                                        
                        else:
                            self.strat[ self.pairs.index(pair)] = 1 
                        
                    elif ( actual_hedge / hedge_ratio ) < 9/10 :
                        
                        buy_st_1 = - st_1_tot - (st_2_tot / hedge_ratio)
                        
                        if ask_st_1.price < self.df.iloc[ self.pairs.index(pair)]['st_1_price']:
                            
                            vol_st_1 = np.floor(min(buy_st_1, limit - pos[st_1]))
                            
                            if vol_st_1 != 0:
                                
                                st_1_id = self.e.insert_order(st_1, price = ask_st_1.price, volume = int(vol_st_1), side='bid', order_type='ioc')
                                
                                if st_1_id:
                                    
                                    tr_1_vol = self.e.get_positions()[st_1] - pos[st_1]
                                    vwap = self.df.loc[ self.pairs.index(pair)]['st_1_price']
                                    vol = self.df.loc[ self.pairs.index(pair)]['st_1_vol']
                                    prof = (tr_1_vol * ( self.df.iloc[ self.pairs.index(pair)]['st_1_price'] - ask_st_1.price))
                                    self.profits.loc[st_1_id] = ( [pair, st_1, tr_1_vol, vwap, ask_st_1.price, prof] )
                                    self.df.at[ self.pairs.index(pair), 'st_1_vol'] += tr_1_vol
                                    
                                    if tr_1_vol + vol == 0:
                                        
                                        self.df.at[ self.pairs.index(pair), 'st_1_price'] = 0
                                        
                        else:
                            self.strat[ self.pairs.index(pair)] = 2
                        
                    else:
                        self.strat[ self.pairs.index(pair)] = 0
                                      
                        
                st_1_tot = self.df.iloc[ self.pairs.index(pair)]['st_1_vol']
                st_2_tot = self.df.iloc[ self.pairs.index(pair)]['st_2_vol']
                
                if st_1_tot != 0 or st_2_tot != 0:
                    
                    if st_1_tot != 0:
                        actual_hedge = - (st_2_tot / st_1_tot)
                        
                    else:
                        actual_hedge = np.nan
                        
                    ratio = actual_hedge / hedge_ratio
                    hedge.loc[ self.pairs.index(pair)] = ( [pair, actual_hedge, hedge_ratio, ratio] )
                    
        return hedge
    
        
    def update_thresholds(self, hist_prof, thresholds, t = 1):
        
        hist_prof.append( self.profits['Profit'].sum() )
        upper_threshold = thresholds[-1]
        
        if t == 1:
            
            upper_threshold += 0.05
            thresholds.append(upper_threshold)
            
        else:
            
            prof_change = hist_prof[-1] - hist_prof[-2]
            
            if  prof_change > 0:
                
                upper_threshold += (prof_change / hist_prof[-2]) * 0.1 * np.sign(thresholds[-1] - thresholds[-2])
                thresholds.append(min(upper_threshold, 2.0))
                
            if prof_change < 0:
                
                upper_threshold += (prof_change/hist_prof[-2]) * 0.1 * np.sign(thresholds[-1] - thresholds[-2])
                thresholds.append(max(upper_threshold, 0.4))
                
        return hist_prof, thresholds    
        
        
        
    def start_trading(self):

        
        a = 0
        
        while True:
            
            try:
                
                a += 1
                
                if a % 10 == 0:
                    
                    if a % 100 != 0:
                        self.hedge_ratio_diff()
                        
                if a % 100 == 0:
                    
                    clear_output(wait=True)
                    logger.info('')
                    print('Profits:')
                    print(self.profits.tail(10))
                    print('Hedging:')
                    print(self.hedge_ratio_diff())
                    print('Positions:')
                    print(self.df)
                    
                if a % 10000 == 0:
                    
                    if a == 10000:
                        
                        self.hist_prof, self.thresholds = self.update_thresholds(self.hist_prof, self.thresholds, t = 1)
                        self.profits = self.profits.iloc[0:0]
                        
                    else:
                        
                        self.hist_prof, self.thresholds = self.update_thresholds(self.hist_prof, self.thresholds, t = 0)
                        self.profits = self.profits.iloc[0:0]
                        
                self.trade(self.thresholds)
                time.sleep(0.1)    
            
            
            except Exception as e:
                
                clear_output(wait=True)
                print(e)
                self.e.disconnect()
                time.sleep(1)
                self.e.connect()
                time.sleep(1)
                self.get_out_of_positions()
                
                continue 
                
        
if '__main__' == __name__:
    
    data = pd.read_csv('cointegration/data.csv')
    bot = statistical_arbitrage(data)
    bot.start_trading()
        
        
        
        