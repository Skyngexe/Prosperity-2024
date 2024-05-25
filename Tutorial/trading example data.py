from datamodel import Listing, OrderDepth, Position, Trade, TradingState

timestamp = 1100

listings = {
    "PRODUCT1": Listing(
        symbol="PRODUCT1", product="PRODUCT1", denomination="SEASHELLS"
    ),
    "PRODUCT2": Listing(
        symbol="PRODUCT2", product="PRODUCT2", denomination="SEASHELLS"
    ),
}

order_depths = {
    "PRODUCT1": OrderDepth(buy_orders={10: 7, 9: 5}, sell_orders={12: -5, 13: -3}),
    "PRODUCT2": OrderDepth(buy_orders={142: 3, 141: 5}, sell_orders={144: -5, 145: -8}),
}

own_trades = {
    "PRODUCT1": [
        Trade(
            symbol="PRODUCT1",
            price=11,
            quantity=4,
            buyer="SUBMISSION",
            seller="",
            timestamp=1000,
        ),
        Trade(
            symbol="PRODUCT1",
            price=12,
            quantity=3,
            buyer="SUBMISSION",
            seller="",
            timestamp=1000,
        ),
    ],
    "PRODUCT2": [
        Trade(
            symbol="PRODUCT2",
            price=143,
            quantity=2,
            buyer="",
            seller="SUBMISSION",
            timestamp=1000,
        ),
    ],
}

market_trades = {"PRODUCT1": [], "PRODUCT2": []}

position = {"PRODUCT1": 10, "PRODUCT2": -7}

observations = {}

traderData = ""

state = TradingState(
    traderData,
    timestamp,
    listings,
    order_depths,
    own_trades,
    market_trades,
    position,
    observations,
)


# for product in state.order_depths:
#       order_depth: OrderDepth = state.order_depths[product]
#       print("Order here: " + str(order_depth))

for product in state.position:
    # position: Position = state.position[product]
    own_trades["PRODUCT1"][0].quantity  # should give 4
    position2: Position = state.position.get(product, 0)
    print("Position here: " + str(position2))

star = list(position.keys())
for i in range(len(star)):
    # position: Position = state.position[product]
    owntrades2 = own_trades[star[i]][0].quantity  # should give 4
    position2: Position = state.position.get(star[i], 0)
    print("Position here: " + str(position2))
    print("own trades here: " + str(owntrades2))
