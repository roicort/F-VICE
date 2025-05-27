
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