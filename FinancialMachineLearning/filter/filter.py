import matplotlib.pyplot as plt
import pandas as pd


def plot_filter(raw_time_series, t_pos_events, t_neg_events):
    plt.figure(figsize=(16, 12))
    ax = raw_time_series.plot()
    ax.scatter(t_pos_events, raw_time_series.loc[t_pos_events], color='red')
    ax.scatter(t_neg_events, raw_time_series.loc[t_neg_events], color='black')
    plt.title("CUSUM filtered events")
    plt.show()


def cusum_filter(raw_time_series, threshold, time_stamps=True, plot=False):
    t_events = []
    t_pos_events = []
    t_neg_events = []
    s_pos = 0
    s_neg = 0
    diff = raw_time_series.diff()
    for i in diff.index[1:]:
        pos = s_pos + diff.loc[i]
        neg = s_neg + diff.loc[i]
        s_pos = max(0, pos)
        s_neg = min(0, neg)
        if s_neg < -threshold:
            s_neg = 0
            t_pos_events.append(i)
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_neg_events.append(i)
            t_events.append(i)

    if plot:
        plot_filter(raw_time_series, t_pos_events, t_neg_events)

    if time_stamps:
        t_pos_events = pd.DatetimeIndex(t_pos_events)
        t_neg_events = pd.DatetimeIndex(t_neg_events)
        return t_pos_events, t_neg_events

    return t_pos_events, t_neg_events


def z_score_filter(raw_time_series, mean_window, std_window, z_score=3, time_stamps=True, plot=False):
    rolling_mean = raw_time_series.rolling(window=mean_window).mean()
    rolling_std = raw_time_series.rolling(window=std_window).std()

    condition_upper = raw_time_series >= (rolling_mean + z_score * rolling_std)
    condition_lower = raw_time_series <= (rolling_mean - z_score * rolling_std)

    t_pos_events = raw_time_series[condition_upper].index
    t_neg_events = raw_time_series[condition_lower].index
    
    z_score_filter = pd.DataFrame(0, index=raw_time_series.index, columns=['value'])
    z_score_filter.loc[t_pos_events] = 1
    z_score_filter.loc[t_neg_events] = -1
    
    if plot:
        plot_filter(raw_time_series, t_pos_events, t_neg_events)

    if time_stamps:
        t_pos_events = pd.DatetimeIndex(t_pos_events)
        t_neg_events = pd.DatetimeIndex(t_neg_events)
        return  t_pos_events, t_neg_events, z_score_filter

    return t_pos_events, t_neg_events, z_score_filter
