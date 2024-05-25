# Prosperity_year2
If you push new code, please notify in the Discord first.

## Installing the Pre-Push Hook
We use a custom Git pre-push hook in our project to ensure code pushed to the main branch has been properly updated and rebased. This section guides you through setting up this hook.

### 1 - Pull the Latest Changes
Make sure you have the latest version of the repository:
```commandline
git pull origin main
```

### 2 - Run the Setup Script
Execute the setup_hooks.sh script to install the pre-push hook:
```commandline
sh git_hooks/setup_hooks.sh
```

### 3 - Run the Algo
Do inspec on your IMC prosperity page, and check for CognitoIdentityServiceProvider...idToken. Copy to your clipboard and then execute the following command
```commandline
pbpaste | poetry run prosperity2submit Tutorial/Trader.py
```


# Algorithmic Trading Challenge: SeaShells Trader

## Introduction
Welcome to the SeaShells Trader Challenge! In this competition, you'll develop a Python-based trading algorithm to compete against bots in a simulated island exchange environment. The aim is to accrue as many SeaShells, the island's virtual currency, as you can through strategic trading.

## How It Works
Your task is to create a `Trader` class that contains your trading logic. This class interacts with a variety of other classes provided within the simulation environment, which together facilitate your interaction with the market.

### 1. Trader Class
`Trader` is your main class where the algorithm resides.
- **run**: The heart of your strategy, which processes the `TradingState` to make trading decisions.
- **result**: A dictionary output of your orders, keyed by product.
- **traderData**: A string to maintain state across the algorithm's execution calls.

### 2. TradingState Class
Provides a snapshot of the current market situation.
- **Attributes**: Includes trader data, timestamp, listings, order depths, trades, position data, and observations.

### 3. Order and Execution Mechanics
Describes how orders are placed and executed within the simulation.

### 4. OrderDepth Class
Details outstanding market orders, influencing trading decisions.

### 5. Trade Class
Represents an individual trade, providing insights into market activities.

### 6. Order Class
Defines a trading order's structure.

### 7. Listing Class
Contains metadata for tradable products.

### 8. Observation Class
Encapsulates market insights and conditions impacting trading decisions.

### 9. ConversionObservation Class
Offers data relevant to product conversions, essential for strategies involving asset conversion.

### 10. Placing Orders
Describes how the algorithm should place orders using the `Order` class.

## Strategy Tips
- **Understanding Position Limits**: Manage your exposure to products, and strategize within the set position limits.
- **Enhancing Trading Strategies**: Utilize the provided data classes to tailor strategies, exploit opportunities, and adapt to market conditions.

## Getting Started
1. Clone this repository.
2. Navigate to the cloned directory.
3. Implement your `Trader` class logic.
4. Test your algorithm using the simulation environment.
5. Submit your `Trader` class as instructed within the challenge platform.