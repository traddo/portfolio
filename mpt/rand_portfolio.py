import numpy as np
import pandas as pd
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt


def rand_weight(n):
    k = np.random.rand(n)
    return(k / sum(k))


def rand_portfolio(returns):
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weight(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    if(sigma > 2):
        return rand_portfolio(returns)
    return mu, sigma


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
    pd_mus_data = pd.DataFrame(mus)

    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


np.random.seed(123)
solvers.options['show_progress'] = False

n_assets, n_obs = 4, 1000
return_vec = np.random.randn(n_assets, n_obs)


'''
plt.plot(return_vec.T[:100],alpha=.8)
plt.xlabel('time')
plt.ylabel('returns')
plt.grid(linestyle=':')
plt.show()
'''

n_portfolios = 1000
means, stds = np.column_stack([rand_portfolio(return_vec)
                               for _ in range(n_portfolios)])
weights, returns, risks = optimal_portfolio(return_vec)
print(weights)

plt.plot(stds, means, 'o', markersize=3)
plt.plot(risks, returns, 'y-o', markersize=3)
plt.show()
