import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
    UserId,
)

# storing string as const to avoid typos
SUBMISSION = "SUBMISSION"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"

PRODUCTS = [
    AMETHYSTS,
    STARFRUIT,
]

DEFAULT_PRICES = {
    AMETHYSTS: 10_000,
    STARFRUIT: 5_000,
}


class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            AMETHYSTS: 20,
            STARFRUIT: 20,
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
        self.amethysts_prices = (
            []
        )  # List to store historical AMETHYSTS prices for calculation
        self.log_return = []

        self.ema_param = 0.5
        # self.df_log_return = pd.DataFrame([], columns=pd.Index(['Log_Return']))

    # utils
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    # Duck attempt at adding microprice
    def get_micro_price(self, product, state):
        """
        Compute microprice based on best bid and ask data.
        Microprice = [best bid price * {best ask volume / (best bid volume + best ask volume)}] +
                 [best ask price * {best bid volume / (best bid volume + best ask volume)}]
        If microprice < midprice then potentially sell at best available (or midprice).
        If microprice > midprice then potentially buy at best available (or midprice).
        """
        if product not in state.order_depths:
            return None  # Handle case where product is not in the order depths

        order_book = state.order_depths[product]

        # Assuming that the order book entries are tuples of (price, volume)
        if not order_book.sell_orders or not order_book.buy_orders:
            return None  # Handle case where there are no bids or asks

        # Using max/min on order book assuming they're sorted by price (dicts of price: volume)
        best_ask_price, best_ask_volume = min(
            order_book.sell_orders.items(), key=lambda x: x[0]
        )
        best_bid_price, best_bid_volume = max(
            order_book.buy_orders.items(), key=lambda x: x[0]
        )

        # Calculate total volume
        total_volume = best_bid_volume + best_ask_volume

        # Avoid division by zero
        if total_volume == 0:
            return None

        # Calculate microprice
        microprice = best_bid_price * (
            best_ask_volume / total_volume
        ) + best_ask_price * (best_bid_volume / total_volume)

        return microprice

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

    def amethysts_strategy(self, state: TradingState):
        """
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
        print(f"best ask: {best_ask}\n")
        print(f"best bid: {best_bid}\n")
        orders.append(
            Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 2, bid_volume)
        )  # buy order
        orders.append(
            Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 2, ask_volume)
        )  # sell order
        return orders
        """
        microprice = self.get_micro_price(AMETHYSTS, state)
        mid_price = self.get_mid_price(AMETHYSTS, state)
        position_amethysts = self.get_position(AMETHYSTS, state)
        bid_volume = self.position_limit[AMETHYSTS] - position_amethysts
        ask_volume = -self.position_limit[AMETHYSTS] - position_amethysts

        orders = []
        print("Is micrpice ", microprice is not None)
        print(" is mid price ", mid_price is not None)
        if microprice is not None and mid_price is not None:
            print(f"| Mid Price: {mid_price} |")
            print(f"| Microprice: {microprice} |")
            print(f"| ask volume: {ask_volume} |")
            print(f"| bid volume: {bid_volume} |")
            if microprice < mid_price:
                # Strategy to sell
                orders.append(Order(AMETHYSTS, mid_price, ask_volume))
            elif microprice > mid_price:
                # Strategy to buy
                orders.append(Order(AMETHYSTS, mid_price, bid_volume))
        return orders

    # strat for startfruit
    def starfruit_strategy(self, state: TradingState):
        """
        best_ask, _ = next(
            iter(state.order_depths[STARFRUIT].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[STARFRUIT].buy_orders.items()), (None, None)
        )
        orders = []
        if best_ask and best_bid:
            mid_price = (best_bid + best_ask) / 2
            ## Duck
            microprice = self.get_micro_price(STARFRUIT, state)

            self.starfruit_prices.append(mid_price)
            print(f"| Mid Price: {mid_price} |")
            print(f"| Microprice: {microprice} |")
            log_return = np.log(
                mid_price / self.starfruit_prices[len(self.starfruit_prices) - 2]
            )
            print(f"| Log Return: {log_return} |")
            print(f"| past price: {self.starfruit_prices[-1]} |")
            # self.log_return.append(log_return)
            position_starfruit = self.get_position(STARFRUIT, state)
            bid_volume = self.position_limit[STARFRUIT] - position_starfruit
            ask_volume = -self.position_limit[STARFRUIT] - position_starfruit
            if log_return > 0.025:
                orders.append(Order(STARFRUIT, int(mid_price), ask_volume))
            elif log_return < 0:
                orders.append(Order(STARFRUIT, int(mid_price), bid_volume))
        return orders
        """
        microprice = self.get_micro_price(STARFRUIT, state)
        mid_price = self.get_mid_price(STARFRUIT, state)
        position_starfruit = self.get_position(STARFRUIT, state)
        bid_volume = self.position_limit[STARFRUIT] - position_starfruit
        ask_volume = -self.position_limit[STARFRUIT] - position_starfruit

        orders = []
        print("Is micrpice ", microprice is not None)
        print(" is mid price ", mid_price is not None)
        if microprice is not None and mid_price is not None:
            print(f"| Mid Price: {mid_price} |")
            print(f"| Microprice: {microprice} |")
            print(f"| ask volume: {ask_volume} |")
            print(f"| bid volume: {bid_volume} |")
            if microprice < mid_price:
                # Strategy to sell
                orders.append(Order(STARFRUIT, mid_price, ask_volume))
            elif microprice > mid_price:
                # Strategy to buy
                orders.append(Order(STARFRUIT, mid_price, bid_volume))
        return orders

    #  if log_return>0.05:
    #         orders.append(Order(STARFRUIT,int(mid_price), ask_volume))
    #       elif log_return < 0:
    #         orders.append(Order(STARFRUIT, int(mid_price), bid_volume))

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

        # Initialize the method output dict as an empty dict
        result = {}

        # AMETHYST STRATEGY
        try:
            result[AMETHYSTS] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        # STARFRUIT STRATEGY
        try:
            result[STARFRUIT] = self.starfruit_strategy(state)
        except Exception as e:
            print("Error in starfruit strategy")
            print(e)

        print("+---------------------------------+")

        return result, None, None
