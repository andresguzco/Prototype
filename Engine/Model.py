import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from Engine.Tools import *
from scipy.stats import t
import numpy as np


class Engine:
    def __init__(self,
                 input_df: pd.DataFrame,
                 ) -> None:
        self.forecast = None
        self.data = input_df
        self.results = None
        self.lam_ext_up = None
        self.lam_up = None
        self.lam_down = None
        self.lam_ext_down = None
        self.mean_model = None
        self.vol_model = None
        self.shock = None
        self.metric = None

    def run(self) -> None:
        self.get_params()
        # self.metric = SMAPE(actual=self.data['Client rate'].diff(1).iloc[1:], forecast=self.mean_model.fittedvalues)
        # self.get_shocks()
        self.estimate_lambda()
        return None

    def get_shocks(self) -> None:aNDRES       alpha_mod = sm.tsa.AutoReg(
            endog=self.metric,
            lags=1,
            trend='c').fit()
        print(alpha_mod.summary())
        self.shock = np.std(alpha_mod.resid) / (1 - alpha_mod.params[0] ** 2)
        return None

    def print_results(self) -> None:
        print(self.mean_model.summary())
        print(f'The extreme shock up scalar is: {self.lam_ext_up:.2f}')
        print(f'The shock up scalar is: {self.lam_up:.2f}')
        print(f'The shock down scalar is: {self.lam_down:.2f}')
        print(f'The extreme shock down scalar is: {self.lam_ext_down:.2f}')
        return None

    def estimate_lambda(self) -> None:
        self.lam_ext_up = self.shock * t.ppf(q=0.001, df=len(self.data) - 4)
        self.lam_up = self.shock * t.ppf(q=0.012, df=len(self.data) - 4)
        self.lam_down = self.shock * t.ppf(q=0.988, df=len(self.data) - 4)
        self.lam_ext_down = self.shock * t.ppf(q=0.999, df=len(self.data) - 4)
        return None

    def plot_fitted(self, portfolio: str) -> None:
        estimates = self.data["Client rate"].iloc[1:] + self.mean_model.fittedvalues
        plt.plot(self.data["Client rate"].sort_index().iloc[1:], label="Client Rate")
        plt.plot(self.data["Short rate"], label="SR")
        plt.plot(estimates, label="Model")
        plt.title(f'{portfolio}: Fitted Values')
        plt.legend()
        plt.show()
        return None

    def plot_errors(self, portfolio: str) -> None:
        obs = self.metric
        plt.plot(obs, label="Model")
        plt.title(f'{portfolio}: Residuals')
        plt.legend()
        plt.show()
        return None

    def get_params(self) -> None:
        self.run_WLS()
        self.run_GARCH()
        self.vol_model.plot()
        plt.plot()
        print(self.vol_model.summary())
        self.results = self.mean_model.params
        self.results = self.mean_model.fittedvalues
        return None

    def run_GARCH(self) -> None:
        endog = self.mean_model.resid
        self.mean_model = arch_model(
            y=endog,
            mean="Constant",
            vol='GARCH',
            p=1,
            o=1,
            q=1,
            dist='t',
            hold_back=None,
            rescale=None
        ).fit()
        self.forecast = self.mean_model.forecast(horizon=12, reindex=False)
        print(self.forecast)
        return None

    def run_WLS(self) -> None:
        self.mean_model = sm.WLS(
            endog=self.data["Client rate"].iloc[1:],
            exog=self.data[["Client rate", "Short rate", "Steepness"]].iloc[:-1],
            weights=self.data["Volume"].iloc[:-1]
        ).fit()
        print(self.mean_model.summary())
        return None


def MAPE(actual: pd.DataFrame, forecast: pd.DataFrame) -> np.ndarray:
    n = len(actual)
    mape = np.zeros(n)
    for i in range(n):
        if np.abs(actual.iloc[i]) > 0.001:
            mape[i] = np.abs(forecast.iloc[i] - actual.iloc[i]) / np.abs(actual.iloc[i])
        else:
            mape[i] = 0
    return mape


def SMAPE(actual: pd.DataFrame, forecast: pd.DataFrame) -> np.ndarray:
    n = len(actual)
    smape = np.zeros(n)
    for i in range(n):
        smape[i] = np.abs(forecast.iloc[i]-actual.iloc[i]) / ((np.abs(forecast.iloc[i])+np.abs(actual.iloc[i]))/2)
    return smape
