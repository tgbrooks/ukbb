import pandas
import numpy

def read_actigraphy(file):
    data = pandas.read_csv(file, sep=",", index_col=0, parse_dates=[0])

    data = data.set_index(index)
    data.rename(columns={data.columns[0]:"acceleration"})# Acceleration column name contains some extra information, ignore it

    # clear the imputed values since they are junk
    is_imputed = data.imputed.astype(bool)
    data.loc[is_imputed] = float("NaN")
    data[is_imputed, 'imputed'] = 1 # restore the 'imputed' value since we just overwrote it as NaN

    return data
