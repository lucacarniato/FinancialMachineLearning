import numpy as np
import pandas as pd
def cusum_filter(raw_time_series, threshold, time_stamps=True):
    t_events = []
    s_pos = 0
    s_neg = 0
    diff = np.log(raw_time_series).diff()
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)
        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps
    return t_events