import io
import os
import zipfile
import pandas as pd


def save_df_to_zip(df, fname, index=False, **kwargs):
    assert isinstance(df, pd.DataFrame)
    fname, ext = os.path.splitext(fname)
    df_str = df.to_csv(index=index, **kwargs)
    bytes_io = io.BytesIO()
    with zipfile.ZipFile(bytes_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(fname + ext, df_str)
    with open(fname + '.zip', 'wb') as f:
        f.write(bytes_io.getvalue())
