from statsmodels.api import WLS, add_constant
from typing import List, Union, Any

from numpy import ndarray, percentile, random, std
from Engine.ModelTypes import GARCHModel
import matplotlib.pyplot as plt
from pandas import DataFrame


class Engine:
    def __init__(
            self,
            input_df: DataFrame
            ) -> None:
        self.data = input_df
        self.simulation_df = None
        self.mean_model = None
        self.vol_model = None
        self.shocks = None

    def run(self) -> None:
        self.data["Gap"] = self.data["Short rate"] - self.data["Client rate"]
        self.data["Indicator"] = self.data["Gap"].apply(lambda x: 1 if x > 0 else 0)
        self.run_WLS()
        self.estimate_shock_GARCH()
        return None

    def estimate_shock_GARCH(self) -> None:
        self.vol_model = GARCHModel(data=(self.mean_model.resid / (1-self.mean_model.params[1])))
        self.vol_model.fit()
        self.shocks = self.vol_model.estimate_alpha_VaR(
            alphas=[0.001, 0.012, 0.988, 0.999],
            num_simulations=1000000
        )
        return None

    def estimate_shock_Benchmark(self) -> None:
        self.shocks = shock_sampling(
            alphas=[0.001, 0.012, 0.988, 0.999],
            num_simulations=1000000,
            WLS_errors=self.mean_model.resid
        )
        return None

    def run_WLS(self) -> None:
        endog = self.data["Client rate"].iloc[1:].reset_index(drop=True)
        exog = add_constant(
            self.data[["Client rate", "Short rate", "Steepness", "Indicator"]].iloc[:-1].reset_index(drop=True)
        )
        w = self.data["Volume"].iloc[:-1].reset_index(drop=True)

        self.mean_model = WLS(
            endog=endog,
            exog=exog,
            weights=w
        ).fit(cov_type='HC1', use_t=True)
        return None

    def print_results(self) -> None:
        print(self.mean_model.summary())
        print(self.vol_model.model.summary())
        print(f'The extreme shock up scalar is: {self.shocks[0]:.2f}')
        print(f'The shock up scalar is: {self.shocks[1]:.2f}')
        print(f'The shock down scalar is: {self.shocks[2]:.2f}')
        print(f'The extreme shock down scalar is: {self.shocks[3]:.2f}')
        return None

    def plot_fitted(self, portfolio: str) -> None:
        plt.plot(self.data["Client rate"].sort_index(), label="Client Rate")
        plt.plot(self.data.index[1:], self.mean_model.fittedvalues, label="Model")
        plt.plot(self.data["Short rate"], label="SR")
        plt.title(f'{portfolio}: Fitted Values')
        plt.legend()
        plt.show()
        return None

    def plot_errors(self, portfolio: str) -> None:
        plt.plot(self.mean_model.resid, label="Model")
        plt.title(f'{portfolio}: Residuals')
        plt.legend()
        plt.show()
        return None


def shock_sampling(
        alphas=None,
        num_simulations: int = 10000,
        WLS_errors: ndarray = None
) -> List[Union[Union[int, float, complex], Any]]:

    if alphas is None:
        alphas = [0.10, 0.05, 0.01]

    random.seed(123)
    simulated_errors = random.normal(loc=0, scale=std(WLS_errors), size=num_simulations)

    output = list()
    for i, alpha in enumerate(alphas):
        output.append(-percentile(simulated_errors, alpha * 100))
    return output
