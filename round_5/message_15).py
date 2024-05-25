import math
from typing import Any, Dict, List, Tuple,Union
import itertools
import numpy as np
import pandas as pd
from datamodel import Order, TradingState

# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"
STRAWBERRIES="STRAWBERRIES"
ROSES="ROSES"
CHOCOLATE="CHOCOLATE"
GIFT_BASKET="GIFT_BASKET"
COCONUT = "COCONUT"
COCONUT_COUPON = "COCONUT_COUPON"

PRODUCTS = [
    AMETHYSTS,
    STARFRUIT, 
    ORCHIDS,
    STRAWBERRIES,
    ROSES,
    CHOCOLATE,
    GIFT_BASKET,
    COCONUT,
    COCONUT_COUPON
]

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
    ORCHIDS: 1090,
    STRAWBERRIES: 4000,
    ROSES:15000,
    CHOCOLATE:8000,
    GIFT_BASKET:71355,
    COCONUT: 1000,
    COCONUT_COUPON:500
}

POSITION_LIMITS = {
  STRAWBERRIES: 350,
  ROSES:60,
  CHOCOLATE:250,
  GIFT_BASKET:60,
  COCONUT_COUPON:600,
  COCONUT:300
}

WINDOW = 20
VOLUME_BASKET = 2

import numpy as np

class AdaptiveTrendFiltering:
    def __init__(self, order=1, regularization=1.0):
        self.order = order
        self.regularization = regularization
        self.beta = None
        self.x_points = None  # Store x_points after fitting

    def _falling_factorial_basis(self, x, x_points):
        n = len(x_points)
        m = len(x)
        # The basis should have n columns because the h_{k+1,j}(x) basis functions go up to n - k - 1
        basis = np.zeros((m, n))

        # Compute the first k basis functions h_j(x)
        for j in range(1, self.order + 1):
            h_j = np.ones(m)
            for ell in range(j-1):
                h_j *= (x - x_points[ell])
            basis[:, j-1] = h_j

        # Compute the k+1-th basis functions h_{k+1,j}(x)
        for j in range(1, n - self.order):
            h_k1_j = np.ones(m)
            for ell in range(self.order):
                h_k1_j *= (x - x_points[j + ell - 1])
            # Apply the step function conditionally
            h_k1_j *= (x >= x_points[j + self.order - 1])
            basis[:, self.order + j - 1] = h_k1_j

        return basis

    def _kth_order_difference_matrix(self, size):
        """
        Create a (k+1)-th order difference matrix D of size (size-k-1) x size for k-th order differencing.
        """
        k = self.order
        D = np.eye(size)
        for _ in range(k+1):
            D = np.diff(D, n=1, axis=0)
        return D

    def compute_basis_matrix(self, x):
        """
        Compute the falling factorial basis matrix H for the given input x based on the x_points.
        """
        if self.x_points is None:
            raise ValueError("x_points must be set before calling compute_basis_matrix.")
        
        H = self._falling_factorial_basis(x, self.x_points)
        return H

    def fit(self, x, y):
        """
        Fit the model using Ridge regression on k-th order differences.
        """
        self.x_points = np.sort(np.array(x))  # Sort x_points for basis function construction
        H = self.compute_basis_matrix(self.x_points)
        D = self._kth_order_difference_matrix(H.shape[1])

        # Construct the regularized matrix
        regularized_matrix = H.T @ H + self.regularization * D.T @ D
        # Solve for the coefficients
        self.beta = np.linalg.solve(regularized_matrix, H.T @ y)
        return H @ self.beta
    
    def forecast(self, x_new):
        """
        Forecast future values for new input data x_new.
        :param x_new: The new input values for which to predict outputs.
        :return: Predicted values for the input x_new.
        """
        if self.beta is None:
            raise Exception("Model must be fitted before forecasting.")

        # Check if x_points was set during fitting
        if self.x_points is None:
            raise Exception("x_points were not set during fitting. Cannot forecast.")

        # Compute the basis matrix for the new x values
        H_new = self.compute_basis_matrix(x_new)

        # Calculate the forecasted values
        y_new = H_new @ self.beta
        return y_new
    

class Trader:
    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            AMETHYSTS : 20,
            STARFRUIT : 20,
            ORCHIDS: 100,
            COCONUT_COUPON:600,
            COCONUT:300
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position

        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.starfruit_prices = []  # List to store historical STARFRUIT prices for calculation
        self.amethysts_prices = []
        self.orchid_prices = []   # List to store historical ORCHID prices for calculation
        self.starfruit_log_return = []
        self.amethysts_log_return = []
        self.orchid_log_return = []
        self.spread_gift_list = []
        self.chocolate_past_price = []
        self.rose_past_price = []
        self.strawberries_past_price = []
        self.basket_past_price =[]
        self.past_prices_gift_spread = {product: pd.Series(dtype='float64') for product in PRODUCTS}
        self.ema_param = 0.5
        self.timestamp = []
        self.basket_adjustment_factor_array = []
        self.strawberries_adjustment_factor_array = []
        self.coconut_price = []
        self.prices : Dict[str, pd.Series] = {
            "Spread":pd.Series(),            
            "SPREAD_PICNIC": pd.Series(),
            "SPREAD_GIFT": pd.Series()
        }
        self.ratio = []

        self.conversion_amt = 0
        # self.df_log_return = pd.DataFrame([], columns=pd.Index(['Log_Return']))


    # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2

    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price

        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value

        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]
              
    def save_prices_product(
      self, 
      product, 
      state: TradingState,
      price: Union[float, int, None] = None, 
    ):
      if not price:
          price = self.get_mid_price(product, state)

      self.prices[product] = pd.concat([
          self.prices[product],
          pd.Series({state.timestamp: price})
      ])

    # finalized I guess 
    def amethysts_strategy(self, state: TradingState):

      position_amethysts = self.get_position(AMETHYSTS, state)

      bid_volume = self.position_limit[AMETHYSTS] - position_amethysts
      ask_volume = -self.position_limit[AMETHYSTS] - position_amethysts

      print(f"Position Amethysts: {position_amethysts}")
      print(f"Bid Volume: {bid_volume}")
      print(f"Ask Volume: {ask_volume}")

      orders = []
      best_ask, _ = next(
          iter(state.order_depths[AMETHYSTS].sell_orders.items()), (None, None)
      )
      best_bid, _ = next(
          iter(state.order_depths[AMETHYSTS].buy_orders.items()), (None, None)
      )
      mid_price = (best_bid + best_ask) / 2
      self.amethysts_prices.append(mid_price)
      log_return = np.log(mid_price/self.amethysts_prices[len(self.amethysts_prices)-2])
      #self.amethysts_log_return.append(log_return)
      #print(f"best ask: {best_ask}\n")
      #print(f"best bid: {best_bid}\n")
      orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 2, bid_volume)) 
      orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 2, ask_volume))  
      return orders



#strat for startfruit (finalized)
    def starfruit_strategy(self, state : TradingState):
        best_ask, _ = next(iter(state.order_depths[STARFRUIT].sell_orders.items()), (None, None))
        best_bid, _ = next(iter(state.order_depths[STARFRUIT].buy_orders.items()), (None, None))
        orders = []
        if best_ask and best_bid:
          mid_price = (best_bid + best_ask) / 2
          self.starfruit_prices.append(mid_price)
          #print(f"| Mid Price: {mid_price} |") 
          log_return = np.log(mid_price/self.starfruit_prices[len(self.starfruit_prices)-2])
          #print(f"| Log Return: {log_return} |")
          #print(f"| past price: {self.starfruit_prices[-1]} |")
          # self.log_return.append(log_return)
          position_starfruit = self.get_position(STARFRUIT, state)
          bid_volume = self.position_limit[STARFRUIT] - position_starfruit
          ask_volume = -self.position_limit[STARFRUIT] - position_starfruit
          if log_return> 0.05: 
            orders.append(Order(STARFRUIT,int(mid_price + 2), ask_volume)) # try comment these out 
            orders.append(Order(STARFRUIT, int(mid_price - 1), bid_volume)) 
          elif log_return < 0:
            orders.append(Order(STARFRUIT,int(mid_price + 1), ask_volume)) # try comment these out 
            orders.append(Order(STARFRUIT, int(mid_price - 2), bid_volume))
        return orders


    def Orchid_strategy(self, state: TradingState):
        best_ask, _ = next(iter(state.order_depths[ORCHIDS].sell_orders.items()), (None, None))
        best_bid, _ = next(iter(state.order_depths[ORCHIDS].buy_orders.items()), (None, None))
        #print(f"best ask: {best_ask}")
        #print(f"best bid: {best_bid}")
        orders = []
        position_orchid = self.get_position(ORCHIDS, state)

        bid_volume = self.position_limit[ORCHIDS] - position_orchid
        ask_volume = -self.position_limit[ORCHIDS] - position_orchid

        orchid_conversion = state.observations.conversionObservations[ORCHIDS]
        south_archipelago_bid_price = orchid_conversion.bidPrice
        south_archipelago_ask_price = orchid_conversion.askPrice 
        shipping_cost = orchid_conversion.transportFees
        import_tariff = orchid_conversion.importTariff
        export_tariff = orchid_conversion.exportTariff

        buy_price_from_south_archipelago = south_archipelago_ask_price + shipping_cost + import_tariff
        sell_price_to_south_archipelago = south_archipelago_bid_price + shipping_cost + export_tariff

        if best_ask and best_bid:
            mid_price = (best_ask + best_bid)/2 
            self.orchid_prices.append(mid_price)
            log_returns = []
            for _ in range(1, len(self.orchid_prices)):
                log_return = np.log(mid_price/ self.orchid_prices[len(self.orchid_prices)-2]) 
                log_returns.append(log_return)
            if len(log_returns) > 0:
                average_log_return = np.mean(log_returns)
            else:
                average_log_return = 0

            optimal_quantity = (mid_price * average_log_return) / 0.1 # change it to the new version 
            amount_wanted = optimal_quantity - position_orchid

            buy_quantity = int(max(amount_wanted, bid_volume))
            sell_quantity = int(min(-amount_wanted, ask_volume))
            #print("buy and sell quantity:", buy_quantity, sell_quantity)
            #print("avg log return and midprice: ", average_log_return,mid_price)
            sell_on_our_archipelago = average_log_return * mid_price * (optimal_quantity + sell_quantity) + (-sell_quantity * mid_price) - 0.1*(optimal_quantity + sell_quantity)
            sell_to_south_archipelago = average_log_return * mid_price *(optimal_quantity + sell_quantity) + (-sell_quantity*buy_price_from_south_archipelago) -(0.1*(optimal_quantity+ sell_quantity))
            buy = average_log_return * mid_price * (optimal_quantity + buy_quantity) - (buy_quantity*mid_price) -(0.1*(optimal_quantity + buy_quantity))
            buy_from_south_archipelago = average_log_return * -sell_price_to_south_archipelago * (optimal_quantity + buy_quantity) - (buy_quantity*-sell_price_to_south_archipelago) -(0.1*(optimal_quantity + buy_quantity))
            #fuck_it_we_ball = mid_price*optimal_quantity*average_log_return - (0.1*(optimal_quantity + buy_quantity))
            decision = max(sell_on_our_archipelago, sell_to_south_archipelago, -buy,  buy_from_south_archipelago)
            print(sell_on_our_archipelago, sell_to_south_archipelago, -buy, buy_from_south_archipelago)

            if decision != 0:
                if decision == sell_on_our_archipelago:
                    print("sell_on_our_archipelago") # if we are selling, we are selling at the best_bid 
                    orders.append(Order(ORCHIDS, best_bid, sell_quantity)) # sell on island, buy from south 
                    #self.conversion_amt = - sell_quantity
                elif decision == sell_to_south_archipelago:
                    print("ff")
                    #self.conversion_amt = sell_quantity
                elif decision == buy:  
                    orders.append(Order(ORCHIDS, best_ask, buy_quantity))
                    #self.conversion_amt = buy_quantity
                elif decision == buy_from_south_archipelago:
                    self.conversion_amt = buy_quantity
                    orders.append(Order(ORCHIDS, int(mid_price-1), -buy_quantity))
                    print("buy from south: ", self.conversion_amt)
                    print("order: ", orders)

        return orders



    def update_past_prices_gift_spread(self, spread: float):
        """
        Update past prices of gift spread and calculate rolling statistics.
        """
        if 'spread' not in self.past_prices_gift_spread:
            self.past_prices_gift_spread['spread'] = []
        self.past_prices_gift_spread['spread'].append(spread)

        if len(self.past_prices_gift_spread['spread']) > WINDOW:
            # Update rolling statistics
            spread_data = self.past_prices_gift_spread['spread'][-WINDOW:]
            avg_spread = np.mean(spread_data)
            std_spread = np.std(spread_data)
            self.past_prices_gift_spread['avg'] = avg_spread
            self.past_prices_gift_spread['std'] = std_spread

    
    def coconut_predictive_strategy(self, state: TradingState):
        coconut_orders = []
        coconut_coupon = []
        forecast_steps = 5
        coconut_price = self.get_mid_price(COCONUT, state)
        coconut_coupon_price = self.get_mid_price(COCONUT_COUPON, state)
        self.coconut_price.append(coconut_price)
        #x_data = state.timestamp
        #y_data = self.coconut_price
        self.timestamp.append(state.timestamp)
        coco_position_coconut = self.get_position(COCONUT, state)
        coupon_position_coconut = self.get_position(COCONUT, state)
        buy_quantity = self.position_limit[COCONUT] - coco_position_coconut
        sell_quantity = -self.position_limit[COCONUT] - coco_position_coconut

        coupon_buy_quantity = self.position_limit[COCONUT_COUPON] - coupon_position_coconut
        coupon_sell_quantity = -self.position_limit[COCONUT_COUPON] - coupon_position_coconut
        steps = 5
        model_order = 3
        model_reg = 0.001
        model = AdaptiveTrendFiltering(order=1, regularization=model_reg)
        if len(self.coconut_price) > 400:
            x_train, x_test = self.timestamp[:-steps], self.timestamp[-steps:]
            y_train, y_test = self.coconut_price[:-steps], self.coconut_price[-steps:]
            # Fit the model to the data and get the estimated y_hat
            y_hat_train = model.fit(x_train, y_train)
            y_hat_test = model.forecast(x_test)
            if y_hat_test[0] > coconut_price:
                coconut_orders.append(Order(COCONUT, int(coconut_price - 1), buy_quantity)) # buying coconut 
                coconut_coupon.append(Order(COCONUT_COUPON, int(coconut_coupon_price + 1), coupon_sell_quantity)) # sell a call option 
            else:
                coconut_orders.append(Order(COCONUT, int(coconut_price + 1), sell_quantity))
                coconut_coupon.append(Order(COCONUT_COUPON, int(coconut_coupon_price - 1), coupon_buy_quantity))

            print(y_hat_test)
        return coconut_orders, coconut_coupon
    
    def gb_strategy(self, state: TradingState)-> Tuple[List[Order], List[Order], List[Order], List[Order]]:
      """Gift strategy. Trades on spread between gift basket and
      6*strawberries+ 4*chocolate+1*rose
      """        
      orders_chocolate = []
      orders_strawberries = []
      orders_roses = []
      orders_basket = []
     
      price_basket = self.get_mid_price(GIFT_BASKET, state)
      price_strawberries = self.get_mid_price(STRAWBERRIES, state)
      price_roses = self.get_mid_price(ROSES, state)
      price_chocolate = self.get_mid_price(CHOCOLATE, state) 
      position_basket = self.get_position(GIFT_BASKET, state)
      position_strawberries = self.get_position(STRAWBERRIES, state)
      predicted_basket_price,actual_predicted_basket_price = 0,0
      self.chocolate_past_price.append(price_chocolate)
      self.rose_past_price.append(price_roses)
      self.strawberries_past_price.append(price_strawberries)
      self.basket_past_price.append(price_basket)
      
      if len(self.chocolate_past_price) > 1:
        predicted_basket_price = (-25.97 + 0.18 * self.chocolate_past_price[-1] 
        - 0.17 * self.chocolate_past_price[-2] + 0.05 * self.rose_past_price[-1] - 0.05 * self.rose_past_price[-2] 
        + 0.15 * self.strawberries_past_price[-1] - 0.11 * self.strawberries_past_price[-2] 
        + 0.97 * self.basket_past_price[-1] - 0.03 * self.basket_past_price[-2])  

        predicted_strawberries_price = 19.8 + 0.97*self.chocolate_past_price[-1]  +0.024*self.chocolate_past_price[-2]
        - 0.002*self.rose_past_price[-1] +0.002*self.rose_past_price[-2] 
        - 0.002*self.strawberries_past_price[-1] + 0.02*self.strawberries_past_price[-2]
        +0.003* self.basket_past_price[-1] - 0.004* self.basket_past_price[-2]
         # randomly select three timestamp, (12.98)
        # find difference between the actual price and predicted price, 12.98 = (difference) / 3 (+ 4050)
        # self.chocolate_past_price[-1] --> price_chocolate| for rose, strawberries
        # self.chocolate_past_price[-2] --> self.chocolate_past_price[-1]
        adjustment_factor_basket = price_basket-predicted_basket_price 
        adjustment_factor_strawberries = price_strawberries - predicted_strawberries_price
        #df_day_0['chocolate_varest']=19.8+0.97*df_day_0.CHOCOLATE.shift(1)+0.024*df_day_0.CHOCOLATE.shift(2)
        #-0.002*df_day_0.ROSES.shift(1)+0.002*df_day_0.ROSES.shift(2)
        #-0.002*df_day_0.STRAWBERRIES.shift(1)+0.02*df_day_0.STRAWBERRIES.shift(2)+0.003*df_day_0.GIFT_BASKET.shift(1)
        #-0.004*df_day_0.GIFT_BASKET.shift(2)


        if len(self.basket_adjustment_factor_array) < 50:
            self.basket_adjustment_factor_array.append(adjustment_factor_basket)
            self.strawberries_adjustment_factor_array.append(adjustment_factor_strawberries)
        else:
            self.basket_adjustment_factor_array.pop(0)
            self.strawberries_adjustment_factor_array.pop(0)
            self.basket_adjustment_factor_array.append(adjustment_factor_basket)
            self.strawberries_adjustment_factor_array.append(adjustment_factor_strawberries)

        avg_adjustment_factor_basket = np.mean(self.basket_adjustment_factor_array) 
        avg_adjustment_factor_strawberries = np.mean(self.strawberries_adjustment_factor_array)
        actual_predicted_basket_price = predicted_basket_price+ avg_adjustment_factor_basket
        actual_predicted_strawberries_price = predicted_strawberries_price+ avg_adjustment_factor_strawberries
        #print("actual_predicted_basket_price: ", actual_predicted_basket_price, price_basket, avg_adjustment_factor_basket)
        VOLUME_BASKET = 60 - position_basket
        VOLUME_STRAWBERRIES = 350 - position_strawberries
        if actual_predicted_basket_price > price_basket:
           orders_basket.append(Order(GIFT_BASKET, int(price_basket-1),  VOLUME_BASKET)) # buy
        elif actual_predicted_basket_price < price_basket:
           orders_basket.append(Order(GIFT_BASKET, int(price_basket+1),  -1 * VOLUME_BASKET)) # sell

    #   if actual_predicted_strawberries_price > price_strawberries:
    #       orders_strawberries.append(Order(STRAWBERRIES, int(price_strawberries-1), VOLUME_STRAWBERRIES))
    #   elif actual_predicted_strawberries_price < price_strawberries:
    #       orders_strawberries.append(Order(STRAWBERRIES, int(price_strawberries+1),  -1 * VOLUME_STRAWBERRIES)) # sell

      #4020+0.18df_day_0.CHOCOLATE.shift(1)-0.17df_day_0.CHOCOLATE.shift(2)+0.05df_day_0.ROSES.shift(1)
      # -0.05df_day_0.ROSES.shift(2)+0.15df_day_0.STRAWBERRIES.shift(1)
      #-0.11df_day_0.STRAWBERRIES.shift(2)+0.97df_day_0.GIFT_BASKET.shift(1)-0.03df_day_0.GIFT_BASKET.shift(2)

      return orders_basket
    
    # finalized 
    def bunny_gift_basket_strategy(self, state: TradingState) -> List[Order]:
      orders_basket = []
      orders_roses = []
      orders_choco = []
      orders_strawberries = []
      try:
        # Calculate the spread between GIFT_BASKET and its components
        #gift_basket_price = self.get_mid_price('GIFT_BASKET',state)
        #strawberries_price = self.get_mid_price('STRAWBERRIES',state)
        #roses_price = self.get_mid_price('ROSES',state)
        #chocolate_price = self.get_mid_price('CHOCOLATE',state) 

        price_basket = self.get_mid_price(GIFT_BASKET, state)
        price_strawberries = self.get_mid_price(STRAWBERRIES, state)
        price_roses = self.get_mid_price(ROSES, state)
        price_chocolate = self.get_mid_price(CHOCOLATE, state) 
        position_basket = self.get_position(GIFT_BASKET, state)
        spread = price_basket - (price_roses*1 + (price_chocolate * 4) + (price_strawberries * 6))
        #print(f"| price_basket: {price_basket} |")
        #print(f"| price_strawberries: {price_strawberries} |")
        #print(f"| price_roses: {price_roses} |")
        #print(f"| price_chocolate: {price_chocolate} |")
        #print(f"| spread: {spread} |")

        self.spread_gift_list.append(spread)
        serie = pd.Series(self.spread_gift_list)

        if len(self.spread_gift_list) < WINDOW:
          avg_spread = serie.rolling(len(self.prices["SPREAD_GIFT"])).mean()
          std_spread = serie.rolling(len(self.prices["SPREAD_GIFT"])).std()
        else:
          std_spread = serie.rolling(WINDOW).std()
          avg_spread = serie.rolling(WINDOW).mean()
        if len(self.spread_gift_list) < 5:
          spread_5 = serie.rolling(len(self.prices["SPREAD_GIFT"])).mean()
        else:
          spread_5= serie.rolling(5).mean()


        if not np.isnan(avg_spread.iloc[-1]):
          avg_spread = avg_spread.iloc[-1]
          std_spread = std_spread.iloc[-1]
          spread_5 = spread_5.iloc[-1]
          
      
          # Calculate Z-score of the spread
          z_score = (spread - avg_spread) / std_spread
          print(f"spread: {spread}, mean_spread: {avg_spread}, std_dev_spread:{std_spread}")
          print(f"Z-score: {z_score}")
        
          # Define the threshold for Z-score to trigger trading
          z_score_threshold = 1.5
       
          # Make trading decision based on Z-score
          if z_score > z_score_threshold:       
            # Sell GIFT_BASKET and buy components
            orders_basket.append(Order('GIFT_BASKET', int(price_basket), -3)) # int(price_basket)
            orders_strawberries.append(Order('STRAWBERRIES', int(price_strawberries), 6)) #18
            orders_roses.append(Order('ROSES', int(price_roses), 1)) # 3 
            orders_choco.append(Order('CHOCOLATE', int(price_chocolate), 4)) #12
            
          elif z_score  < z_score_threshold:
            # Buy GIFT_BASKET and sell components
              orders_basket.append(Order('GIFT_BASKET', int(price_basket), 3)) 
              orders_strawberries.append(Order('STRAWBERRIES', int(price_strawberries), -6))
              orders_roses.append(Order('ROSES', int(price_roses), -1))
              orders_choco.append(Order('CHOCOLATE', int(price_chocolate), -4))
          
      except Exception as e:
        print("Error in gift basket strategy:")
        print(e)

      return orders_choco, orders_basket, orders_strawberries, orders_roses
    
    def coconut_and_coupon_bollinger_strategy(self, state: TradingState):
        orders_coconut = []
        orders_coconut_coupon = []
        
        coconut_best_ask, _ = next(iter(state.order_depths[COCONUT].sell_orders.items()), (None, None))
        coconut_best_bid, _ = next(iter(state.order_depths[COCONUT].buy_orders.items()), (None, None))

        coconut_coupon_best_ask, _ = next(iter(state.order_depths[COCONUT_COUPON].sell_orders.items()), (None, None))
        coconut_coupon_best_bid, _ = next(iter(state.order_depths[COCONUT_COUPON].buy_orders.items()), (None, None))
        
        COCONUT_current_price = self.get_mid_price(COCONUT, state)
        COCONUT_COUPON_current_price = self.get_mid_price(COCONUT_COUPON, state)

        print("mid prices: ", COCONUT_current_price, COCONUT_COUPON_current_price)
       
        position_coconut = self.get_position(COCONUT, state)
        position_coconut_coupon = self.get_position(COCONUT_COUPON, state)

        print("positions: ", position_coconut, position_coconut_coupon)
        

        if COCONUT_current_price and COCONUT_COUPON_current_price is None:
            return orders_coconut, orders_coconut_coupon
        
        ratio = COCONUT_current_price/COCONUT_COUPON_current_price
        if ratio is not None:
            # Calculate volume to invest (full volume)
            coconut_volume = self.position_limit[COCONUT] - position_coconut
            coconut_coupon_volume = self.position_limit[COCONUT_COUPON] - position_coconut_coupon
            #print("COCONUT and COCONUT_COUPON vol: ",coconut_volume,coconut_coupon_volume )
            self.ratio.append(ratio)
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(state, ratio)
            #print("ratio, upper_band, lower_band",ratio, upper_band, lower_band )
            # Buy at lower band
            if ratio < lower_band: # long COCONUT, short COCONUT_COUPON
                orders_coconut.append(Order(COCONUT, int(COCONUT_current_price), coconut_volume))
                orders_coconut_coupon.append(Order(COCONUT_COUPON, int(COCONUT_COUPON_current_price), -coconut_coupon_volume))
            
            # Sell at upper band
            if ratio > upper_band:
                orders_coconut.append(Order(COCONUT, int(COCONUT_current_price), -coconut_volume))
                orders_coconut_coupon.append(Order(COCONUT_COUPON, int(COCONUT_COUPON_current_price), coconut_coupon_volume))

        return orders_coconut, orders_coconut_coupon

    def calculate_bollinger_bands(self, state: TradingState, product: str, window=20, num_std_dev=2.5):
        price_series = self.ratio
        
        if len(price_series) < window:
            return None, None, None
        
        rolling_mean = np.mean(price_series[-window:])
        rolling_std = np.std(price_series[-window:])
        
        upper_band = rolling_mean + num_std_dev * rolling_std
        lower_band = rolling_mean - num_std_dev * rolling_std
        
        return upper_band, rolling_mean, lower_band

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.conversion_amt = 0
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
        print(f"Log round {self.round}")

        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        print(f"\tPnL {pnl}")

        #check if pnl has hit 107,860
        # self.below_110 = False
        # if pnl >= 107860:
        #     print("pnl has hit 110,000, not executing further trades")
        #     self.below_110 = False
        #     return {}, None, None
        # elif pnl < 107860:
        #     self.below_110 = True


        # Initialize the method output dict as an empty dict
        result = {}
        
        # PEARL STRATEGY

        # try:
        #     result[AMETHYSTS] = self.amethysts_strategy(state)
        # except Exception as e:
        #     print("Error in amethysts strategy")
        #     print(e)
        
        # # BANANA STRATEGY
        # try:
        #     result[STARFRUIT] = self.starfruit_strategy(state)
        # except Exception as e:
        #     print("Error in starfruit strategy")
        #     print(e)

        # print("+---------------------------------+")
        # try:
        #     result[ORCHIDS] = self.Orchid_strategy(state)
        #     position = self.get_position(ORCHIDS, state) 
        #     if position != 0 :
        #         if position > 0:
        #             self.conversion_amt = position
        #         else:
        #             self.conversion_amt = - position
        #         print("orchids: " ,self.get_position(ORCHIDS, state))
        #         print("conversions: ", self.conversion_amt)
        # except Exception as e:
        #     print("Error in orchid strategy")
        #     print(e)

        #print("+---------------------------------+")
        # try:
        #     result[COCONUT], \
        #     result[COCONUT_COUPON] = self.coconut_and_coupon_bollinger_strategy(state)
        # except Exception as e:
        #     print("Error in coconut_and_coupon_bollinger_strategy") 
        #     print(e)
        
        try:
            result[COCONUT], result[COCONUT_COUPON] = self.coconut_predictive_strategy(state)
        except Exception as e:
            print("Error in coconut_predictive_strategy") 
            print(e)


        # BUNNY_STRAT
        # try:
        #     result[CHOCOLATE], \
        #     result[GIFT_BASKET], \
        #     result[STRAWBERRIES], \
        #     result[ROSES] = self.bunny_gift_basket_strategy(state)
        # except Exception as e:
        #     print("Error in gift basket strategy")
        #     print(e)

        # try:
        #    result[GIFT_BASKET] = self.gb_strategy(state)

        # except Exception as e:
        #   print("Error in gift basket strategy") 
        #   print(e)

        # # #combined strat 

    
        return result, self.conversion_amt, None