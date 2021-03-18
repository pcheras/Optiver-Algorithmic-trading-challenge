from optibook.synchronous_client import Exchange
import time
import logging
logger = logging.getLogger('client')
logger.setLevel('INFO')


class dual_listing:
    
    instruments = ["PHILIPS_A", "PHILIPS_B"]
    
    def __init__(self):
        self.e = Exchange()
        logging.info(self.e.connect())
        logging.info("Setup was successful.")

    def get_out_of_positions(self):
        # Get out of all positions you are currently holding, regardless of the loss involved. That means selling whatever
        # you are long, and buying-back whatever you are short. Be sure you know what you are doing when you use this logic.
        print(self.e.get_positions())
        for s, p in self.e.get_positions().items():
            if p > 0:
                self.e.insert_order(s, price=1, volume=p, side='ask', order_type='ioc')
            elif p < 0:
                self.e.insert_order(s, price=100_000, volume=-p, side='bid', order_type='ioc')  
        print(self.e.get_positions())
    
    # Logging functions
    
    def log_new_trade_ticks(self):
        logger.info("Polling new trade ticks")
        for i in self.instruments:
            tradeticks = self.e.poll_new_trade_ticks(i)
            for t in tradeticks:
                logger.info(f"[{t.instrument_id}] price({t.price}), volume({t.volume}), aggressor_side({t.aggressor_side}), buyer({t.buyer}), seller({t.seller})")
    
    def log_positions_cash(self):
        logger.info(self.e.get_positions_and_cash())
    
    def log_all_outstanding_orders(self):
        for i in self.instruments:
            logger.info(self.e.get_outstanding_orders(i))
    
    def wait_until_orders_complete(self):
        orders_outstanding = True
        while orders_outstanding:
            orders_outstanding = False
            for i in self.instruments:
                if len(self.e.get_outstanding_orders(i)) > 0:
                    orders_outstanding = True
            self.log_all_outstanding_orders()
            #time.sleep(0.1)
    
    def trade(self):
        
        while True:
            
            books = [self.e.get_last_price_book(x) for x in self.instruments]
            A_ask , B_bid = books[0].asks[0] , books[1].bids[0] # Ask book for Philips_A and bid book for Philips_B
            A_bid , B_ask = books[0].bids[0] , books[1].asks[0] # Other way around
            
            try:
                
                if B_ask.price < A_bid.price:
                
                    volume = min(B_ask.volume , A_bid.volume)
                    logger.info(f"Sell {self.instruments[0]} at {A_bid.price} and Buy {self.instruments[1]} at {B_ask.price}")
                    self.e.insert_order(self.instruments[0] , price = A_bid.price , volume = volume , side='ask', order_type='ioc') # Sell A
                    self.e.insert_order(self.instruments[1] , price = B_ask.price , volume = volume , side = 'bid', order_type = 'ioc') # Buy B (hedge in the more liquid order book)
                    
                    self.log_all_outstanding_orders()
                    self.wait_until_orders_complete()
                    self.log_positions_cash()
                    
                    
                elif A_ask.price < B_bid.price:
                
                    volume = min(A_ask.volume , B_bid.volume)
                    logger.info(f"Buy {self.instruments[0]} at {A_bid.price} and Sell {self.instruments[1]} at {B_ask.price}")
                    self.e.insert_order(self.instruments[0] , price = A_ask.price , volume = volume , side='bid', order_type='ioc') # Buy A
                    self.e.insert_order(self.instruments[1] , price = B_bid.price , volume = volume , side = 'ask', order_type = 'ioc') # Sell B (hedge in the more liquid order book)
                    
                    self.log_all_outstanding_orders()
                    self.wait_until_orders_complete()
                    self.log_positions_cash()
                    
            except Exception as e:
                print(logger.error(e))
                continue
            
            time.sleep(0.02)
            
            
            
if __name__ == '__main__':
    
    bot = dual_listing()
    bot.trade()