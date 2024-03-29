import pandas as pd


def format_data(df: pd.DataFrame, segment: str) -> [dict or pd.DataFrame]:
    output_dict = {}
    if segment == 'Country':
        values = df[segment].unique()
        for value in values:
            output_dict[value] = clean_data(df[df[segment] == value].copy())
        return output_dict
    elif segment == 'Portfolio':
        return clean_data(df=df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    output = df.drop(columns={'Portfolio ID', 'Business unit', 'Product', 'Segment', 'Incentive',
                              'Country', 'Volume', "Observable"}).copy()
    return output


def import_data(path: str, type_res: str = "Equilibrium") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="CBR_EC")
    df = df[df["Observable"] == type_res]
    df.dropna(axis=0, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.set_index("Date", inplace=True, drop=True)
    df["Steepness"] = df["Long rate"] - df["Short rate"]
    return df
