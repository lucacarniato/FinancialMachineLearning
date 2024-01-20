import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numba as nb

def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[:: -1]).reshape(-1, 1)
    return w

@nb.jit(nopython=True)
def getWeights_FFD(d, thres):
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[::-1])[1:]
    return w

@nb.jit(nopython=True)
def dot_product(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@nb.jit(nopython=True)
def fracDiff_FFD_array(series_array, w):
    width = len(w) - 1
    result = np.empty(series_array.shape[0])
    for iloc1 in range(width, series_array.shape[0]):
        start_index = iloc1 - width
        end_index = iloc1 + 1
        if not np.isfinite(series_array[start_index]):
            continue
        result[iloc1] = dot_product(w, series_array[start_index: end_index])
    return result

def fracDiff_FFD(series, d, thres=1e-5):
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for column_name in series.columns:
        current_series = series[column_name].ffill().dropna()
        series_array = current_series.to_numpy(dtype=np.float32)
        frac_series_array = fracDiff_FFD_array(series_array, w)
        df[column_name] = pd.Series(frac_series_array[width:], index = current_series.index[width:])
    df = pd.concat(df, axis=1)
    return df

def getWeights_FFD_old(d, thres):
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[:: -1]).reshape(-1, 1)[1:]
    return w


def fracDiff_FFD_old(series, d, thres=1e-5):
    w = getWeights_FFD_old(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]

        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def fracDiff(series, d, thres=.01):
    w = getWeights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]

            test_val = series.loc[loc, name]
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any():
                continue
            try:
                df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def plot_d(series, num_d = 10):
    possible_d = np.divide(range(1, num_d), num_d)
    tau = 1e-4
    original_adf_stat_holder_all = []

    for i in range(len(possible_d)):
        ts_diff = fracDiff_FFD(series, possible_d[i], tau)
        original_adf_stat_holder = []

        for name in ts_diff.columns:
            adf_p_value = adfuller(ts_diff[name].values)[1]
            original_adf_stat_holder.append(adf_p_value)

        original_adf_stat_holder_all.append(original_adf_stat_holder)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, name in enumerate(series.columns):
        ax.plot(possible_d, [stat[i] for stat in original_adf_stat_holder_all], label=name)

    ax.axhline(y=0.01, color='r')
    ax.set_title('ADF P-value by differencing order in the original series')
    ax.set_xlabel('Differencing Order')
    ax.set_xticks(possible_d)
    ax.set_ylabel('ADF P-value')
    ax.legend()
    plt.show()
