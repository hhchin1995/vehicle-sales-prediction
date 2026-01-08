import pandas as pd

def load_rpt(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        na_values=["NULL", ""]
    )
    return df

if __name__=="__main__":
    df = load_rpt(r'./data/DatiumTrain.rpt')
    print(df.head(5))