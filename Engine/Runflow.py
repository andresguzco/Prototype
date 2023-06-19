from Engine.Tools import *
from Engine.Model import *
from tqdm import tqdm


class Runflow:
    def __init__(
            self, path: str,
            verbose: bool,
            plot_fitted: bool,
            plot_err: bool
    ):

        self.output = DataFrame(columns=['Portfolio', 'A', 'B', 'C', 'Scenario'])
        self.plot_fitted = plot_fitted
        self.plot_errors = plot_err
        self.verbose = verbose
        self.filepath = path
        self.models = {}

        self.equilibrium_data = None
        self.portfolio_data = None
        self.portfolios = None
        self.base_data = None

    def run(self) -> None:

        self.portfolio_data = format_data(import_data(self.filepath), segment='Portfolio')
        self.equilibrium_data = read_excel(self.filepath, sheet_name="EqParam")
        self.portfolio_data.rename(columns={"Observed value": "Client rate"}, inplace=True)
        self.portfolios = list(self.portfolio_data["Description"].unique())

        belgianDict = {
            'BB/WB CA': [
                'BCAB Big',
                'BCAS Small',
                'NOCA Notice account',
                'WCAB'
            ],
            'BB/WB Savings': [
                'BSAB Big',
                'BSAS Small',
                'SESAB Big',
                'SESAS Small'
            ],
            'RB CA': [
                'MCAB MCA All',
                'PRIB Privalis All'
            ],
            'RB Savings - Big': [
                'PLIB Big',
                'PSAB Big'
            ],
            'RB Savings - Small': [
                'PLIS Small',
                'PSAS Small'
            ]
        }

        turkishList = [
            "Orange Savings Account / Retail Savings / EUR",
            "Orange Savings Account / Retail Savings / USD"
        ]

        for portfolio in tqdm(self.portfolios):
            if portfolio in turkishList:
                pass
            elif portfolio in belgianDict.keys():
                if self.verbose:
                    print(f'\n{portfolio}')
                sub_portfolios = belgianDict[portfolio]
                for sub_portfolio in sub_portfolios:
                    self.models[portfolio] = self.processor(portfolio=portfolio, subport=sub_portfolio)
                self.printer(portfolio=portfolio)
            else:
                if self.verbose:
                    print(f'\n{portfolio}')
                self.models[portfolio] = self.processor(portfolio=portfolio)
                self.printer(portfolio=portfolio)

        self.output.to_excel(r".\Data\Output.xlsx", index=False)
        return None

    def printer(self, portfolio: str) -> None:

        if self.verbose:
            self.models[portfolio].print_results()
        if self.plot_fitted:
            self.models[portfolio].plot_fitted(portfolio=portfolio)
        if self.plot_errors:
            self.models[portfolio].plot_errors(portfolio=portfolio)

        return None

    def processor(self, portfolio: str, subport: str = None):

        Shock_ext_up, Shock_up, Shock_down, Shock_ext_down, Model = self.get_results(portfolio=portfolio, sub=subport)
        Shock_base = {'Portfolio': subport if subport is not None else portfolio,
                      'A': 0,
                      'B': 0,
                      'C': 0,
                      'Scenario': 0.50}

        self.looper(up=Shock_up, down=Shock_down, base=Shock_base)
        self.output = self.output.append(DataFrame([Shock_ext_up], columns=self.output.columns))
        self.looper(up=Shock_up, down=Shock_down, base=Shock_base)
        self.output = self.output.append(DataFrame([Shock_ext_down], columns=self.output.columns))
        self.looper(up=Shock_up, down=Shock_down, base=Shock_base)

        return Model

    def looper(self,
               up: dict,
               down: dict,
               base: dict
               ) -> None:

        self.output = self.output.append(DataFrame([up], columns=self.output.columns), ignore_index=True)
        self.output = self.output.append(DataFrame([base], columns=self.output.columns), ignore_index=True)
        self.output = self.output.append(DataFrame([down], columns=self.output.columns), ignore_index=True)

    def get_results(self, portfolio: str, sub: str = None) -> tuple:

        key = sub if sub is not None else portfolio
        placeholder = self.portfolio_data[self.portfolio_data["Description"] == portfolio].copy()
        placeholder.drop(labels=['Description'], axis=1, inplace=True)

        mod = Engine(input_df=placeholder)
        mod.run()

        A = mod.mean_model.params[0] / (1 - mod.mean_model.params[1])
        B = mod.mean_model.params[2] / (1 - mod.mean_model.params[1])
        C = mod.mean_model.params[3] / (1 - mod.mean_model.params[1])

        Shock_ext_up = {'Portfolio': key,
                        'A': A * mod.shocks[0] / 100,
                        'B': B * mod.shocks[0] / 100,
                        'C': C * mod.shocks[0] / 100,
                        'Scenario': 0.001}
        Shock_up = {'Portfolio': key,
                    'A': A * mod.shocks[1] / 100,
                    'B': B * mod.shocks[1] / 100,
                    'C': C * mod.shocks[1] / 100,
                    'Scenario': 0.012}
        Shock_down = {'Portfolio': key,
                      'A': A * mod.shocks[2] / 100,
                      'B': B * mod.shocks[2] / 100,
                      'C': C * mod.shocks[2] / 100,
                      'Scenario': 0.988}
        Shock_ext_down = {'Portfolio': key,
                          'A': A * mod.shocks[3] / 100,
                          'B': B * mod.shocks[3] / 100,
                          'C': C * mod.shocks[3] / 100,
                          'Scenario': 0.999}
        return Shock_ext_up, Shock_up, Shock_down, Shock_ext_down, mod
