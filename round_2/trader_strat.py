# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:56:59 2024

@author: Celes
"""


import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datamodel import Order, TradingState

# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"


PRODUCTS = [AMETHYSTS, STARFRUIT, ORCHIDS]

DEFAULT_PRICES = {AMETHYSTS: 10_000, STARFRUIT: 5_000, ORCHIDS: 1090}


class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {AMETHYSTS: 20, STARFRUIT: 20, ORCHIDS: 100}

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
        self.orchid_log_return = []

        self.ema_param = 0.5
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
        print(f"best ask: {best_ask}\n")
        print(f"best bid: {best_bid}\n")
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
            ask_volume = -self.position_limit[STARFRUIT] - position_starfruit
            if log_return > 0.05:
                orders.append(Order(STARFRUIT, int(mid_price + 2), ask_volume))
            elif log_return < 0:
                orders.append(Order(STARFRUIT, int(mid_price + 2), bid_volume))
        return orders

    #  if log_return>0.05:
    #         orders.append(Order(STARFRUIT,int(mid_price), ask_volume))
    #       elif log_return < 0:
    #         orders.append(Order(STARFRUIT, int(mid_price), bid_volume))

    def Orchid_strategy(self, state: TradingState):
        best_ask, _ = next(
            iter(state.order_depths[ORCHIDS].sell_orders.items()), (None, None)
        )
        best_bid, _ = next(
            iter(state.order_depths[ORCHIDS].buy_orders.items()), (None, None)
        )
        orders = []
        position_orchid = self.get_position(ORCHIDS, state)
        # Add in conditional logic for position_limit based on wiki
        bid_volume = self.position_limit[ORCHIDS] - position_orchid
        ask_volume = -self.position_limit[ORCHIDS] - position_orchid
        orchid_conversion = state.observations.conversionObservations[ORCHIDS]
        south_archipelago_bid_price = orchid_conversion.bidPrice
        south_archipelago_ask_price = orchid_conversion.askPrice
        conversions = {}
        # print(f"|south_archipelago_bid_price: {south_archipelago_bid_price} |")
        # print(f"|south_archipelago_ask_price: {south_archipelago_ask_price} |")
        if best_ask and best_bid:
            mid_price = (best_ask + best_bid) / 2
            self.orchid_prices.append(mid_price)
            # Change to average log return
            log_return = np.log(
                mid_price / self.orchid_prices[len(self.orchid_prices) - 2]
            )  # expected log return
            optimal_quantity = (mid_price * log_return) / 0.1
            # amount_wanted= optimal_quantity-position_orchid
            # buy quantity=min(amount_wanted,position_limit)
            # sell quantity=min(amounted_wanted,-position_limit)
            buy_quantity = min(position_orchid, bid_volume)
            sell_quantity = min(position_orchid, -ask_volume)
            sell_on_our_archipelago = (
                log_return * mid_price * (optimal_quantity + sell_quantity)
                + (-sell_quantity * mid_price)
                - 0.1 * (optimal_quantity + sell_quantity)
            )
            sell_to_south_archipelago = (
                log_return * mid_price * (optimal_quantity + sell_quantity)
                + (-sell_quantity * south_archipelago_ask_price)
                - (0.1 * (optimal_quantity + sell_quantity))
            )
            # buy_to_south_archipelago
            buy = (
                log_return * mid_price * (optimal_quantity + buy_quantity)
                - (buy_quantity * mid_price)
                - (0.1 * (optimal_quantity + buy_quantity))
            )
            fuck_it_we_ball = mid_price * optimal_quantity * log_return - (
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
                    conversions = {"ORCHIDS": sell_quantity}
                    print("sell_to_south_archipelago")
                    pass
                elif decision == buy:
                    orders.append(Order(ORCHIDS, best_ask, buy_quantity))
                    print("buy")
                elif decision == fuck_it_we_ball:
                    print("ball it")
                    pass
        return orders, conversions

    # Convert at last time stamp? Figure out conversions either
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

        # PEARL STRATEGY
        """
        try:
            result[AMETHYSTS] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        # BANANA STRATEGY
        try:
            result[STARFRUIT] = self.starfruit_strategy(state)
        except Exception as e:
            print("Error in starfruit strategy")
            print(e)

        print("+---------------------------------+")
        """
        try:
            result[ORCHIDS] = self.Orchid_strategy(state)
        except Exception as e:
            print("Error in orchid strategy")
            print(e)

        print("+---------------------------------+")

        return result, None, None
