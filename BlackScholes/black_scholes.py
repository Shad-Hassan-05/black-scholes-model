import math
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, date
import numpy as np
from tabulate import tabulate

# There is no way to predict the price of stocks using models (if your excluding insider trading).
# The black-scholes model defines what the risk neutral price of the option would be
# assuming some underlying dynamics of the stock.

# Implementation of black-scholes formula, using an example stock
def black_scholes(S, K, T, r, vol):
    d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - (vol * math.sqrt(T))

    # call option price
    c = S * norm.cdf(d1) - (K * math.exp(-r * T) * norm.cdf(d2))

    # put option price
    p = (K * math.exp(-r * T) * norm.cdf(-d2) - S )* norm.cdf(-d1)

    return c, p

def black_scholes_table(tic, expiry, r = 0.045):

    # retrieve stock information from yahoo finance
    ticker = yf.Ticker(tic)
    option = ticker.option_chain(expiry)
    calls = option.calls

    # calculate time till expiry
    t = ((datetime.strptime(expiry, "%Y-%m-%d"))- datetime.now()).days / 365

    # list to store model call/price
    prices = []

    # get stocks last price
    S = ticker.fast_info['last_price']

    # for each strike price use model to calculate put/call prices
    for idx, call in calls.iterrows():
        K = call['strike']
        vol = call['impliedVolatility']

        # skip invalid vols
        if vol is None or vol <= 0 or np.isnan(vol):
            continue

        # calculate prices with model
        call_px, put_px = black_scholes(S, K, t, r, vol)

        # store information
        prices.append({
            "strike": K,
            "call price": float(call_px),
            "put price": float(put_px),
            "vol": vol
        })

    # displays prices and volatility as a table
    table = []

    for row in prices:
        table.append([
            row["strike"],
            round(row["call price"], 4),
            round(row["put price"], 4),
            round(row["vol"], 4)
        ])

    print(tabulate(
        table,
        headers=["Strike", "Call Price", "Put Price", "Volatility"],
        tablefmt="pretty"
    ))


def blkschl_vs_market(tic, expiry, r = 0.045):
    # retrieve stock information from yahoo finance
    ticker = yf.Ticker(tic)
    option = ticker.option_chain(expiry)
    calls = option.calls

    # calculate time till expiry
    t = ((datetime.strptime(expiry, "%Y-%m-%d")) - datetime.now()).days / 365

    # list to store model call/price
    S = ticker.fast_info['last_price']

    # list to store model and market call/price
    prices = []

    # for each strike price use model to calculate put/call prices and store market price
    for idx, call in calls.iterrows():
        K = call['strike']
        vol = call['impliedVolatility']

        # skip invalid vols
        if vol is None or vol <= 0 or np.isnan(vol):
            continue

        # calculate prices with model
        model_call, _ = black_scholes(S, K, t, r, vol)

        # calculate market prices for call
        mid = (call["bid"] + call["ask"]) / 2
        if np.isnan(mid):
            continue
        mkt_call = float(mid)

        # store values
        prices.append({
            "strike": K,
            "model price": float(model_call),
            "market price": float(mkt_call),
            "difference": float(mkt_call - model_call),
            "vol": vol
        })

    # display information as table in terminal
    table = []

    for row in prices:
        table.append([
            row["strike"],
            round(row["vol"], 4),
            round(row["model price"], 4),
            round(row["market price"], 4),
            round(row["difference"], 4)
        ])

    print(tabulate(
        table,
        headers=["Strike", "Volatility", "Model Call Price", "Market Call Price", "Difference"],
        tablefmt="pretty"
    ))

# example stock
if __name__ == "__main__":
    expiry = '2028-01-21'
    tic = 'AAPL'
    blkschl_vs_market(tic, expiry)