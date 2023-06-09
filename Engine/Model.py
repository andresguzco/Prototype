from Engine.ModelTypes import SVModel, GARCHModel
from statsmodels.api import WLS, add_constant
from statsmodels.tsa. api import coint
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import zeros


class Engine:
    def __init__(
            self,
            input_df: DataFrame
            ) -> None:
        self.data = input_df
        self.simulation_df = None
        self.mean_model = None
        self.vol_model = None
        self.profit = None
        self.shocks = None
        self.gap = None

    def run(self) -> None:
        self.profit = zeros(len(self.data) - 1)
        self.profit[:] = (self.data["Short rate"] - self.data["Client rate"]).iloc[1:]
        self.run_WLS()

        phi = 1 - self.mean_model.params[1]
        self.data["Constant"] = self.mean_model.params[0] / phi
        self.data["Tracking SR"] = (self.mean_model.params[2] / phi) * self.data["Short rate"]
        self.data["Tracking ST"] = (self.mean_model.params[3] / phi) * self.data["Steepness"]
        self.data["Equilibrium"] = self.data["Constant"] + self.data["Tracking SR"] + self.data["Tracking ST"]
        self.data["Equilibrium"] /= 100

        self.gap = zeros(len(self.data) - 1)
        self.gap[:] = (self.data["Short rate"] - self.data["Equilibrium"]).iloc[1:]
        self.estimate_shock_II()
        return None

    def estimate_shock_GARCH(self) -> None:
        self.vol_model = GARCHModel(data=(self.mean_model.resid / (1-self.mean_model.params[1])))
        self.vol_model.fit()
        self.shocks, self.simulation_df = self.vol_model.estimate_alpha_VaR(
            alphas=[0.001, 0.012, 0.988, 0.999],
            num_simulations=1000000
        )
        return None

    def estimate_shock_II(self) -> None:
        self.vol_model = SVModel(data=self.gap)
        self.vol_model.fit()
        self.shocks = self.vol_model.estimate_alpha_VaR(
            alphas=[0.001, 0.012, 0.988, 0.999],
            num_simulations=1000
        )
        return None

    def run_WLS(self) -> None:
        print(coint(y0=self.data["Client rate"], y1=self.data["Short rate"], trend='ct', autolag='aic'))
        endog = self.data["Client rate"].iloc[1:].reset_index(drop=True)
        exog = add_constant(self.data[["Client rate", "Short rate", "Steepness"]].iloc[:-1].reset_index(drop=True))
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
        plt.plot(self.data.index[1:], self.gap[:], label='Market Gap')
        plt.plot(self.data.index[1:], self.profit[:], label='Profit Gap')
        plt.plot(self.data.index[1:], self.mean_model.fittedvalues, label="Model")
        plt.plot(self.data["Short rate"], label="SR")
        plt.title(f'{portfolio}: Fitted Values')
        plt.legend()
        plt.show()
        return None

    def plot_errors(self, portfolio: str) -> None:
        plt.plot(self.mean_model.resid, label="Model")
        plt.plot(self.gap, label="Market Gap")
        plt.title(f'{portfolio}: Residuals')
        plt.legend()
        plt.show()
        return None
