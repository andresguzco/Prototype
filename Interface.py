from Engine.Runflow import Runflow
import warnings


def main() -> None:
    filepath = r"Data/CBR_DataBase_Base.xlsx"
    Runflow(path=filepath,
            verbose=False,
            plot_fitted=True,
            plot_err=True).run()
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
