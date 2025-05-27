from its_live.datacube_tools import DATACUBETOOLS as dctools
import pandas as pd

def get_itslive(coords_list, variable="v"):
    dct = dctools()
    dfs = []
    for lat, lon in coords_list:
        ins3xr, ds_point, point_tilexy = dct.get_timeseries_at_point([lon, lat], "4326", variables=[variable])
        if ins3xr is not None and point_tilexy is not None:
            export = ins3xr[
                ["v", "v_error", "vx", "vx_error", "vy", "vy_error", "date_dt", "satellite_img1", "mission_img1"]
            ].sel(x=point_tilexy[0], y=point_tilexy[1], method="nearest")
            df = export.to_dataframe().reset_index()
            df["lat"] = lat
            df["lon"] = lon
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
    
def get_processed_data(df, min_dt=1, max_dt=120):
    if df.empty:
        raise ValueError("DataFrame is empty. No data to process.")
    
    df = df.dropna(subset=["v"]).copy()
    df["mid_date"] = pd.to_datetime(df["mid_date"], utc=True)

    min_dt = pd.Timedelta(days=min_dt)
    max_dt = pd.Timedelta(days=max_dt)
    df = df[(df["date_dt"] >= min_dt) & (df["date_dt"] <= max_dt)].copy()

    df["v"] = df["v"].astype(int)
    df["year"] = df["mid_date"].dt.year
    df["month"] = df["mid_date"].dt.month
    df["dayofyear"] = df["mid_date"].dt.dayofyear

    return df.sort_values(by="mid_date").reset_index(drop=True)