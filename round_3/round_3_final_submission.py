import itertools
import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from datamodel import Order, TradingState

# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
CHOCOLATE = "CHOCOLATE"
GIFT_BASKET = "GIFT_BASKET"


PRODUCTS = [AMETHYSTS, STARFRUIT, ORCHIDS, STRAWBERRIES, ROSES, CHOCOLATE, GIFT_BASKET]

DEFAULT_PRICES = {
    AMETHYSTS: 10_000,
    STARFRUIT: 5_000,
    ORCHIDS: 1090,
    STRAWBERRIES: 4000,
    ROSES: 15000,
    CHOCOLATE: 8000,
    GIFT_BASKET: 71355,
}

POSITION_LIMITS = {
    STRAWBERRIES: 350,
    ROSES: 60,
    CHOCOLATE: 250,
    GIFT_BASKET: 60,
}
WINDOW = 75
VOLUME_BASKET = 2


class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            AMETHYSTS: 20,
            STARFRUIT: 20,
            ORCHIDS: 100,
            CHOCOLATE: 250,
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
        self.starfruit_prices = (
            []
        )  # List to store historical STARFRUIT prices for calculation
        self.amethysts_prices = []
        self.orchid_prices = (
            []
        )  # List to store historical ORCHID prices for calculation
        self.starfruit_log_return = []
        self.amethysts_log_return = []
        self.orchid_log_return = []
        self.spread_gift_list = []
        self.strawberries_price_list = []

        self.past_prices_gift_spread = {
            product: pd.Series(dtype="float64") for product in PRODUCTS
        }
        self.ema_param = 0.5
        self.prices: Dict[str, pd.Series] = {
            "Spread": pd.Series(),
            "SPREAD_PICNIC": pd.Series(),
            "SPREAD_GIFT": pd.Series(),
        }
        # self.df_log_return = pd.DataFrame([], columns=pd.Index(['Log_Return']))

    # utils
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_mid_price(self, product, state: TradingState):

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
        return (best_bid + best_ask) / 2

    def get_value_on_product(self, product, state: TradingState):
        """
        Returns the amount of MONEY currently held on the product.
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state: TradingState):
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

    def update_ema_prices(self, state: TradingState):
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
                self.ema_prices[product] = (
                    self.ema_param * mid_price
                    + (1 - self.ema_param) * self.ema_prices[product]
                )

    # def save_prices_product(self, product, state: TradingState,price: Union[float, int, None] = None, ):
    #   if not price:
    #       price = self.get_mid_price(product, state)

    #   self.prices[product] = pd.concat([
    #       self.prices[product],
    #       pd.Series({state.timestamp: price})
    #   ])

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
        if best_ask is not None and best_bid is not None:
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = 10000
        self.amethysts_prices.append(mid_price)
        # log_return = np.log(mid_price/self.amethysts_prices[len(self.amethysts_prices)-2])
        # self.amethysts_prices.append(mid_price)
        log_return = np.log(
            mid_price / self.amethysts_prices[len(self.amethysts_prices) - 2]
        )
        # self.amethysts_log_return.append(log_return)
        # print(f"best ask: {best_ask}\n")
        # print(f"best bid: {best_bid}\n")
        orders.append(
            Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 2, bid_volume)
        )  # buy order
        orders.append(
            Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 2, ask_volume)
        )  # sell order
        return orders

    # strat for startfruit
    def starfruit_strategy(self, state: TradingState):
        best_ask, _ = next(
            iter(state.order_depths[STARFRUIT].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[STARFRUIT].buy_orders.items()), (None, None)
        )
        orders = []
        if best_ask and best_bid:
            mid_price = (best_bid + best_ask) / 2
            self.starfruit_prices.append(mid_price)
            print(f"| Mid Price: {mid_price} |")
            log_return = np.log(
                mid_price / self.starfruit_prices[len(self.starfruit_prices) - 2]
            )
            print(f"| Log Return: {log_return} |")
            print(f"| past price: {self.starfruit_prices[-1]} |")
            # self.log_return.append(log_return)
            position_starfruit = self.get_position(STARFRUIT, state)
            bid_volume = self.position_limit[STARFRUIT] - position_starfruit
            ask_volume = -self.position_limit[STARFRUIT] + position_starfruit
            if log_return > 0.05:
                orders.append(Order(STARFRUIT, int(mid_price - 2), bid_volume))
            elif log_return < 0:
                orders.append(Order(STARFRUIT, int(mid_price + 2), ask_volume))
        return orders

    def calculate_bollinger_bands(
        self, state: TradingState, product: str, window=20, num_std_dev=2
    ):
        price_series = state.price_history[product]

        if len(price_series) < window:
            return None, None, None

        rolling_mean = np.mean(price_series[-window:])
        rolling_std = np.std(price_series[-window:])

        upper_band = rolling_mean + num_std_dev * rolling_std
        lower_band = rolling_mean - num_std_dev * rolling_std

        return upper_band, rolling_mean, lower_band

    def chocolate_bollinger_strategy(self, state: TradingState):
        orders = []

        best_ask, _ = next(
            iter(state.order_depths[CHOCOLATE].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[CHOCOLATE].buy_orders.items()), (None, None)
        )

        current_price = self.get_mid_price(CHOCOLATE, state)

        if current_price is None:
            return orders

        position_chocolate = self.get_position(CHOCOLATE, state)

        if best_ask is not None and best_bid is not None:
            # Calculate volume to invest (full volume)
            volume = self.position_limit[CHOCOLATE] - position_chocolate

            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                state, CHOCOLATE
            )

            # Buy at lower band
            if current_price < lower_band:
                orders.append(Order(CHOCOLATE, best_bid, volume))

            # Sell at upper band
            if current_price > upper_band:
                orders.append(Order(CHOCOLATE, best_ask, -volume))

        return orders

    def starfruit_ema_strategy(self, state: TradingState):
        orders = []

        best_ask, _ = next(
            iter(state.order_depths[STARFRUIT].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[STARFRUIT].buy_orders.items()), (None, None)
        )

        current_price = self.get_mid_price(STARFRUIT, state)

        if current_price is None:
            return orders

        if self.ema_prices[STARFRUIT] is None:
            self.ema_prices[STARFRUIT] = current_price
        else:
            self.ema_prices[STARFRUIT] = (
                0.5 * current_price + 0.5 * self.ema_prices[STARFRUIT]
            )  # initially: 0.5

        position_starfruit = self.get_position(STARFRUIT, state)
        bid_volume = self.position_limit[STARFRUIT] - position_starfruit
        ask_volume = -self.position_limit[STARFRUIT] - position_starfruit

        if self.ema_prices[STARFRUIT] is not None:
            if current_price > self.ema_prices[STARFRUIT]:
                orders.append(Order(STARFRUIT, best_ask, ask_volume))  # Buy STARFRUIT
            elif current_price < self.ema_prices[STARFRUIT]:
                orders.append(Order(STARFRUIT, best_bid, bid_volume))  # Sell STARFRUIT

        return orders

    def Orchid_strategy(self, state: TradingState):
        best_ask, _ = next(
            iter(state.order_depths[ORCHIDS].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[ORCHIDS].buy_orders.items()), (None, None)
        )
        print(f"best ask: {best_ask}")
        print(f"best bid: {best_bid}")
        orders = []
        position_orchid = self.get_position(ORCHIDS, state)
        # bid_volume = self.position_limit[ORCHIDS] - position_orchid
        # ask_volume = -self.position_limit[ORCHIDS] - position_orchid
        orchid_conversion = state.observations.conversionObservations[ORCHIDS]
        south_archipelago_bid_price = orchid_conversion.bidPrice
        south_archipelago_ask_price = orchid_conversion.askPrice  # fix conversion thing
        conversions = {}
        # print(f"|south_archipelago_bid_price: {south_archipelago_bid_price} |")
        # print(f"|south_archipelago_ask_price: {south_archipelago_ask_price} |")
        if best_ask and best_bid:
            mid_price = (best_ask + best_bid) / 2
            self.orchid_prices.append(mid_price)
            log_returns = []
            for i in range(1, len(self.orchid_prices)):
                log_return = np.log(
                    mid_price / self.orchid_prices[len(self.orchid_prices) - 2]
                )
                log_returns.append(log_return)
            if len(log_returns) > 0:
                average_log_return = np.mean(log_returns)
            else:
                average_log_return = 0
            optimal_quantity = (
                mid_price * average_log_return
            ) / 0.1  # change it to the new version
            amount_wanted = optimal_quantity - position_orchid
            buy_quantity = max(amount_wanted, self.position_limit[ORCHIDS])
            sell_quantity = min(amount_wanted, -self.position_limit[ORCHIDS])
            # amount_wanted= optimal_quantity-position_orchid
            # buy quantity=min(amount_wanted,position_limit)
            # sell quantity=min(amounted_wanted,-position_limit)

            # buy_quantity = min(position_orchid, bid_volume)
            # sell_quantity = max(position_orchid, - ask_volume)
            print("buy and sell quantity:", buy_quantity, sell_quantity)
            print("avg log return and midprice: ", average_log_return, mid_price)
            sell_on_our_archipelago = (
                average_log_return * mid_price * (optimal_quantity + sell_quantity)
                + (-sell_quantity * mid_price)
                - 0.1 * (optimal_quantity + sell_quantity)
            )
            sell_to_south_archipelago = (
                average_log_return * mid_price * (optimal_quantity + sell_quantity)
                + (-sell_quantity * south_archipelago_ask_price)
                - (0.1 * (optimal_quantity + sell_quantity))
            )
            buy = (
                average_log_return * mid_price * (optimal_quantity + buy_quantity)
                - (buy_quantity * mid_price)
                - (0.1 * (optimal_quantity + buy_quantity))
            )
            fuck_it_we_ball = mid_price * optimal_quantity * average_log_return - (
                0.1 * (optimal_quantity + buy_quantity)
            )
            decision = max(
                sell_on_our_archipelago, sell_to_south_archipelago, buy, fuck_it_we_ball
            )
            print(
                sell_on_our_archipelago, sell_to_south_archipelago, buy, fuck_it_we_ball
            )

            if decision != 0:
                if decision == sell_on_our_archipelago:
                    print("sell_on_our_archipelago")
                    orders.append(Order(ORCHIDS, best_bid, sell_quantity))
                elif decision == sell_to_south_archipelago:
                    # Add sell order at south archipelago price
                    # conversion = {ORCHIDS: sell_quantity}
                    # orders.append(Order(ORCHIDS, south_archipelago_ask_price, sell_quantity))
                    # conversions = {"ORCHIDS": sell_quantity}
                    print("sell_to_south_archipelago")
                elif decision == buy:
                    orders.append(Order(ORCHIDS, best_ask, buy_quantity))
                    print("buy")
                elif decision == fuck_it_we_ball:
                    print("ball it")
                    pass
        return orders

    def calculate_rsi(self, state: TradingState, product: str, window=14):
        price_series = state.price_history[product]

        if len(price_series) < window + 1:
            return 50  # Neutral RSI value

        changes = np.diff(price_series)

        gains = np.maximum(0, changes)
        losses = np.maximum(0, -changes)

        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        if avg_loss == 0:
            return 100  # Maximum RSI value

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def chocolate_mean_reversion_strategy(self, state: TradingState):
        best_ask, _ = next(
            iter(state.order_depths[CHOCOLATE].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[CHOCOLATE].buy_orders.items()), (None, None)
        )
        orders = []
        # print("best ask and best bid", best_ask, best_bid)
        current_price = self.get_mid_price(CHOCOLATE, state)
        # if current_price is None:
        #     return orders
        # print("current price: ", current_price)
        position_chocolate = self.get_position(CHOCOLATE, state)  # CHOCOLATE
        # print("position choco: ", position_chocolate) # able to get this, 0
        # Calculate volume to invest (full volume)
        # print("choco position limit: ", self.position_limit[CHOCOLATE]) # expected: 250
        volume = self.position_limit[CHOCOLATE] - position_chocolate
        # print("volume: ", volume) #250
        # Calculate RSI
        rsi = self.calculate_rsi(state, CHOCOLATE)
        # Buy at lower end of the range and oversold RSI
        # if current_price < best_bid and rsi < 30:
        print("current price, best_bid, best_ask", current_price, best_bid, best_ask)
        if rsi < 30:
            orders.append(Order(CHOCOLATE, int(current_price - 1), volume))  # buy it
        # Sell at upper end of the range and overbought RSI
        # if current_price > best_ask and rsi > 70:
        if rsi > 70:
            orders.append(Order(CHOCOLATE, int(current_price + 1), -volume))
            # orders.append(Order('CHOCOLATE', int(price_chocolate), -12))

        print("order: ", orders)
        return orders

    def starfruit_mean_reversion_strategy(self, state: TradingState):
        best_ask, _ = next(
            iter(state.order_depths[STARFRUIT].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[STARFRUIT].buy_orders.items()), (None, None)
        )
        orders = []
        # print("best ask and best bid", best_ask, best_bid)
        current_price = self.get_mid_price(STARFRUIT, state)
        if current_price is None:
            return orders
        # print("current price: ", current_price)
        position_starfruit = self.get_position(STARFRUIT, state)  # STARFRUIT
        print("position starfruit: ", position_starfruit)
        if best_ask is not None and best_bid is not None:
            # Calculate volume to invest (full volume)
            volume = self.position_limit[STARFRUIT] - position_starfruit
            print("volume: ", volume)
            # Calculate RSI
            # rsi = self.calculate_rsi(state, STARFRUIT)
            # Buy at lower end of the range and oversold RSI
            # if current_price < best_bid and rsi < 30:
            if current_price < best_bid:
                orders.append(Order(STARFRUIT, best_bid, volume))

            # Sell at upper end of the range and overbought RSI
            # if current_price > best_ask and rsi > 70:
            if current_price > best_ask:
                orders.append(Order(STARFRUIT, best_ask, -volume))

            print("order: ", orders)
        return orders

    # new
    def gift_basket_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        try:
            # Calculate the spread between GIFT_BASKET and its components
            # gift_basket_price = self.get_mid_price('GIFT_BASKET',state)
            # strawberries_price = self.get_mid_price('STRAWBERRIES',state)
            # roses_price = self.get_mid_price('ROSES',state)
            # chocolate_price = self.get_mid_price('CHOCOLATE',state)

            price_basket = self.get_mid_price(GIFT_BASKET, state)
            price_strawberries = self.get_mid_price(STRAWBERRIES, state)
            price_roses = self.get_mid_price(ROSES, state)
            price_chocolate = self.get_mid_price(CHOCOLATE, state)
            position_basket = self.get_position(GIFT_BASKET, state)
            spread = price_basket - (
                price_roses * 1 + (price_chocolate * 4) + (price_strawberries * 6)
            )
            print(f"| price_basket: {price_basket} |")
            print(f"| price_strawberries: {price_strawberries} |")
            print(f"| price_roses: {price_roses} |")
            print(f"| price_chocolate: {price_chocolate} |")
            print(f"| spread: {spread} |")

            self.spread_gift_list.append(spread)
            self.strawberries_price_list.append(price_strawberries)
            serie = pd.Series(self.spread_gift_list)
            # def strawberries_strat():
            # self.update_ema_prices(state)

            if len(self.spread_gift_list) < WINDOW:
                avg_spread = serie.rolling(len(self.prices["SPREAD_GIFT"])).mean()
                std_spread = serie.rolling(len(self.prices["SPREAD_GIFT"])).std()
            else:
                std_spread = serie.rolling(WINDOW).std()
                avg_spread = serie.rolling(WINDOW).mean()
            if len(self.spread_gift_list) < 5:
                spread_5 = serie.rolling(len(self.prices["SPREAD_GIFT"])).mean()
            else:
                spread_5 = serie.rolling(5).mean()

            if not np.isnan(avg_spread.iloc[-1]):
                avg_spread = avg_spread.iloc[-1]
                std_spread = std_spread.iloc[-1]
                spread_5 = spread_5.iloc[-1]

                # Calculate Z-score of the spread
                z_score = (spread - avg_spread) / std_spread
                print(
                    f"spread: {spread}, mean_spread: {avg_spread}, std_dev_spread:{std_spread}"
                )
                print(f"Z-score: {z_score}")

                # Define the threshold for Z-score to trigger trading
                z_score_threshold = 1.5

                # Make trading decision based on Z-score
                if z_score > z_score_threshold:
                    # Sell GIFT_BASKET and buy components
                    orders.append(Order("GIFT_BASKET", int(price_basket), -3))
                    # orders.append(Order('STRAWBERRIES', int(price_strawberries), 18))
                    # orders.append(Order('ROSES', int(price_roses), 3))
                    # orders.append(Order('CHOCOLATE', int(price_chocolate), 12))

                elif z_score < z_score_threshold:
                    # Buy GIFT_BASKET and sell components
                    orders.append(Order("GIFT_BASKET", int(price_basket), 3))
                # orders.append(Order('STRAWBERRIES', int(price_strawberries), -18))
                # orders.append(Order('ROSES', int(price_roses), -3))
                # orders.append(Order('CHOCOLATE', int(price_chocolate), -12))

        except Exception as e:
            print("Error in gift basket strategy:")
            print(e)
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

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
            print(
                f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}"
            )
        print(f"\tPnL {pnl}")

        # check if pnl has hit 107,860
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

        try:
            result[AMETHYSTS] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        # try:
        #     result[CHOCOLATE] = self.chocolate_bollinger_strategy(state)
        # except Exception as e:
        #     print("Error in choco strategy")
        #     print(e)

        # try:
        #     result[STARFRUIT] = self.starfruit_mean_reversion_strategy(state)
        # except Exception as e:
        #     print("Error in starfruit mean reversion strategy")
        #     print(e)

        # STARFRUIT LOG RETURN STRATEGY
        # try:
        #     result[STARFRUIT] = self.starfruit_strategy(state)
        # except Exception as e:
        #     print("Error in starfruit strategy")
        #     print(e)

        # print("+---------------------------------+")

        # STARFRUIT EMA STRATEGY (final strat for starfruit)
        try:
            result[STARFRUIT] = self.starfruit_ema_strategy(state)
        except Exception as e:
            print("Error in starfruit ema strategy")
            print(e)

        print("+---------------------------------+")

        """
        try:
            result[ORCHIDS] = self.Orchid_strategy(state)
        except Exception as e:
            print("Error in orchid strategy")
            print(e)

        print("+---------------------------------+")
        """
        conversions = 0

        # GIFT_BASKET_STRAT
        try:
            result[GIFT_BASKET] = self.gift_basket_strategy(state)
        except Exception as e:
            print("Error in gift basket strategy")
            print(e)

        # try:
        #   result[CHOCOLATE], \
        #   result[GIFT_BASKET], \
        #   result[STRAWBERRIES], \
        #   result[ROSES] = self.gb_strategy(state)

        # except Exception as e:
        #   print("Error in gift basket strategy")
        #   print(e)

        return result, conversions, None
