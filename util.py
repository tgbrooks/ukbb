import pandas
import numpy

def read_actigraphy(file):
    data = pandas.read_csv(file, sep=",")

    # Data is every 5s from a start time given at the top of the file in the header
    header = data.columns[0]
    _, start_time, end_time, _ = header.split(' - ')
    start_time = pandas.to_datetime(start_time)
    end_time = pandas.to_datetime(end_time)
    runtime = (end_time - start_time).total_seconds()
    index = start_time + pandas.to_timedelta(numpy.arange(0, runtime+1, 5, dtype=int), unit="s")

    data = data.set_index(index)
    data.columns = ["acceleration", "imputed"]

    # clear the imputed values since they are junk
    data.loc[data.imputed.astype(bool), 'acceleration'] = float("NaN")

    return data
