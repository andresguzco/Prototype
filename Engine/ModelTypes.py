from typing import List, Union, Any

from numpy import exp, var, corrcoef, sqrt, zeros, array, abs
from scipy.stats import kurtosis, norm
from scipy.optimize import minimize
from arch import arch_model
import numpy as np


class SVModel(object):
    def __init__(self, data):
        self.data = data
        self.H = 50 * len(data)
        self.estimated_params = None
        self.f = None

    @staticmethod
    def sim_m_SV(e, par):

        omega, beta, sig2f = par[0], exp(par[1])/(1+exp(par[1])), exp(par[2])
        epsilon, eta = e[:, 0], sqrt(sig2f) * e[:, 1]
        H = len(e)
        x, f = zeros(H), zeros(H)

        f[0] = omega / (1 - beta)
        x[0] = exp(f[0] / 2) * epsilon[0]

        for t in range(1, H):
            f[t] = omega + beta * f[t - 1] + eta[t]
            x[t] = exp(f[t] / 2) * epsilon[t]

        xa = abs(x)
        output = array([var(x), kurtosis(x), corrcoef(xa[1:], xa[:-1])[0, 1]])
        return output

    @staticmethod
    def filter_SV(yt, ft, ft1, theta):
        omega, beta, sig2f = theta
        return yt**2 * exp(-ft) + 3 * ft + (ft - omega - beta * ft1)**2 / sig2f

    def fit(self):
        n = len(self.data)
        xa = np.abs(self.data)
        sample_m = np.array([np.var(self.data), kurtosis(self.data), np.corrcoef(xa[1:], xa[:-1])[0, 1]])

        np.random.seed(123)
        e = np.column_stack((norm.rvs(size=self.H), norm.rvs(size=self.H)))

        b, sig2f = 0.2, 0.1
        omega = np.log(np.var(self.data)) * (1 - b)
        par_ini = np.array([omega, np.log(b / (1 - b)), np.log(sig2f)])

        # Perform the optimization
        res = minimize(lambda par: np.mean((self.sim_m_SV(e, par) - sample_m)**2), par_ini, method='BFGS')
        omega_hat = res.x[0]
        beta_hat = exp(res.x[1]) / (1+exp(res.x[1]))
        sig2f_hat = np.exp(res.x[2])
        self.estimated_params = [omega_hat, beta_hat, sig2f_hat]

        self.f = np.zeros(n)
        self.f[0] = np.log(np.var(self.data))

        for t in range(1, n):
            res = minimize(lambda ft: self.filter_SV(self.data[t], ft, self.f[t - 1], self.estimated_params),
                           self.f[t - 1],
                           method='BFGS')
            self.f[t] = res.x

    def estimate_alpha_VaR(self, alphas: list = None, num_simulations: int = 1000):
        if alphas is None:
            alphas = [0.01]
        if self.estimated_params is None or self.f is None:
            raise ValueError("The model has not been fitted yet. Please call the fit() method before estimating VaR.")

        omega_hat, beta_hat, sig2f_hat = self.estimated_params

        np.random.seed(123)
        e_sim = np.column_stack((norm.rvs(size=num_simulations), norm.rvs(size=num_simulations)))

        f_sim = np.zeros(num_simulations)
        f_sim[0] = self.f[-1]
        x_sim = np.zeros(num_simulations)

        for t in range(1, num_simulations):
            eta_t = np.sqrt(sig2f_hat) * e_sim[t, 1]
            f_sim[t] = omega_hat + beta_hat * f_sim[t - 1] + eta_t
            x_sim[t] = np.exp(f_sim[t] / 2) * e_sim[t, 0]

        output = list()
        for i, alpha in enumerate(alphas):
            output.append(-np.percentile(x_sim, alpha * 100))
        return output


class GARCHModel(object):
    def __init__(
            self,
            data: np.ndarray
    ):
        self.endog = data
        self.model = None
        self.estimated_params = None

    def fit(self):
        self.model = arch_model(
            y=self.endog,
            x=None,
            mean='Zero',
            lags=0,
            vol='GARCH',
            p=1,
            o=1,
            q=1,
            power=1.0,
            dist='t',
            hold_back=None,
            rescale=False
        ).fit(disp=False)
        self.estimated_params = self.model.params
        return

    def estimate_alpha_VaR(
            self,
            alphas: list = None,
            num_simulations: int = 1000
            ) -> List[Union[Union[int, float, complex], Any]]:

        if alphas is None:
            alphas = [0.01]
        if self.estimated_params is None:
            raise ValueError("Please call the fit() method before estimating VaR.")

        omega = self.estimated_params['omega']
        alpha = self.estimated_params['alpha[1]']
        beta = self.estimated_params['beta[1]']

        simulated_returns = np.zeros(num_simulations)
        volatilities = np.zeros(num_simulations)
        volatilities[0] = self.model.conditional_volatility[-1]

        np.random.seed(123)
        for t in range(1, num_simulations):
            simulated_returns[t] = np.random.standard_t(df=5) * np.sqrt(volatilities[t-1])
            volatilities[t] = np.sqrt(omega + alpha * simulated_returns[t-1]**2 + beta * volatilities[t-1]**2)

        output = list()
        for i, alpha in enumerate(alphas):
            output.append(-np.percentile(simulated_returns, alpha * 100))
        return output
