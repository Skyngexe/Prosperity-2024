# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:07:33 2024

@author: Celes
"""
import numpy as np
import pandas as pd

rows, cols = 100, 100
num_sim = 10000
# import matplotlib.pyplot as plt

resell_price = 100
resell_price_epsilon = resell_price + 0.0001

profit = np.zeros((rows, cols))


# Shamelessly nabbed from ChatGPT:
def create_upper_triangular_array(n, start=cols):
    # Create an array of 'NaN' values
    triangular_array = np.full((n, n), np.nan)

    # Fill the upper triangular part of the array
    for col in range(n):  # For each column
        for row in range(col + 1):  # Only go up to the diagonal
            # Assign values starting from 'start' and decreasing
            value = start - row
            if value >= 1:  # Ensure values do not go below 1
                triangular_array[row][col] = value
            else:
                break  # Stop filling this column once the value would go below 1

    return triangular_array


# Gets lower price as a triangular matrix, adds one for non-zero prices
seq = range(cols)
res_lower = np.triu(seq)
# * by the number divided itself to force NAN erros for zeros
res_lower = res_lower + 1 * (res_lower / res_lower)

# Same logic as above, but add 2 as always higher

res_higher = create_upper_triangular_array(cols)


# h=plt.hist(np.random.triangular(0,99.999,100, 100000),bins=2000, density=True)
# plt.show()
for x in range(num_sim):
    # Create a random number, we only use one in each loop to represent that it is only simulating one fish
    # rand=np.random.rand()
    # Times resell_price to scale per price. This will need to be modded when we go for actual numbers
    random_array = np.random.triangular(
        0, resell_price, resell_price_epsilon
    ) * np.ones((rows, cols))
    random_array

    binary_lower = 1 * (res_lower > random_array)

    # Will need to double check this logic
    binary_higher = 1 * ((res_higher > random_array) & (binary_lower == 0))

    # Array created so can do array arithmetic
    resell = resell_price * np.ones((rows, cols))

    profit2 = (resell - res_lower) * binary_lower + (
        resell - res_higher
    ) * binary_higher
    # This could probably be more Pythonic
    profit = profit + profit2

# Take average
profit = profit / num_sim

pd.to_csv("profit.csv")
