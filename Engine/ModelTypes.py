from typing import List, Union, Any, Tuple

from numpy import sqrt, zeros, ndarray, random, percentile
from pandas import DataFrame
from arch import arch_model


class GARCHModel(object):
    def __init__(
            self,
            data: ndarray
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
            o=0,
            q=1,
            power=2.0,
            dist='normal',
            hold_back=None,
            rescale=False
        ).fit(disp=False)
        self.estimated_params = self.model.params
        return

    def estimate_alpha_VaR(
            self,
            alphas: list = None,
            num_simulations: int = 1000
    ) -> Tuple[List[Union[Union[int, float, complex], Any]], DataFrame]:

        if alphas is None:
            alphas = [0.01]
        if self.estimated_params is None:
            raise ValueError("Please call the fit() method before estimating VaR.")

        omega = self.estimated_params['omega']
        alpha = self.estimated_params['alpha[1]']
        beta = self.estimated_params['beta[1]']

        simulated_returns = zeros(num_simulations)
        volatilities = zeros(num_simulations)

        n = self.model.conditional_volatility.shape[0]
        errors = zeros(num_simulations)
        errors[:] = random.normal(loc=0, scale=1, size=num_simulations)

        volatilities[0] = self.model.conditional_volatility[n-1]
        simulated_returns[0] = volatilities[0] * errors[0]

        random.seed(123)
        for t in range(1, num_simulations):
            simulated_returns[t] = errors[t] * sqrt(volatilities[t - 1])
            volatilities[t] = sqrt(omega + alpha * simulated_returns[t - 1] ** 2 + beta * volatilities[t - 1] ** 2)

        output = list()
        for i, alpha in enumerate(alphas):
            output.append(-percentile(simulated_returns, alpha * 100))
        return output
