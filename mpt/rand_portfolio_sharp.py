import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt


def portfolio(weights):
    weights = np.array(weights)
    port_returns = np.dot(weights.T, returns_mean)
    port_variance = np.sqrt(
        np.dot(weights.T, np.dot(np.cov(returns), weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


def min_sharpe(weights):
    return -portfolio(weights)[2]


def min_variance(weights):
    return portfolio(weights)[1]


n_assets, n_obs = 4, 500
returns = np.random.randn(n_assets, n_obs)
cov = np.cov(returns)
returns_mean = np.mean(returns, axis=1)

port_returns, port_variance = [], []
for i in range(4000):
    weights = np.random.rand(n_assets)
    weights /= sum(weights)

    port_returns.append(portfolio(weights)[0])
    port_variance.append(portfolio(weights)[1])

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(n_assets))

optv = sco.minimize(min_variance, n_assets *
                    [1. / n_assets, ], method='SLSQP', bounds=bnds, constraints=cons)
opts = sco.minimize(min_sharpe, n_assets *
                    [1. / n_assets, ], bounds=bnds, constraints=cons)

target_returns = np.linspace(
    portfolio(optv['x'])[0], portfolio(optv['x'])[0] + 0.03, 80)
target_variance = []
for tar in target_returns:
    cons2 = ({'type': 'eq', 'fun': lambda x: portfolio(x)[
             0] - tar}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_variance, n_assets *
                       [1. / n_assets, ], method='SLSQP', bounds=bnds, constraints=cons2)
    target_variance.append(res['fun'])
target_variance = np.array(target_variance)

plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and std-dev of returns')

plt.plot(port_variance, port_returns, 'o', markersize=3)
plt.plot(target_variance, target_returns, 'y-', markersize=15)
plt.plot(portfolio(optv['x'])[1], portfolio(optv['x'])[0], 'y*', markersize=20)
plt.plot(portfolio(opts['x'])[1], portfolio(opts['x'])[0], 'r*', markersize=20)

plt.show()
