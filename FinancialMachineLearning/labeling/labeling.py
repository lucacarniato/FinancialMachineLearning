import numpy as np
import pandas as pd
from FinancialMachineLearning.multiprocess.multiprocess import mp_pandas_obj


def apply_pt_sl_on_t1(bars, events, pt_sl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)
    close = bars['bar_close']
    low = bars['bar_low']
    high = bars['bar_high']
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).items():
        side = events_.at[loc, 'side']
        if side == 1:
            min_prices_return = low[loc: vertical_barrier]
            max_prices_return = high[loc: vertical_barrier]
        else:
            min_prices_return = high[loc: vertical_barrier]
            max_prices_return = low[loc: vertical_barrier]

        cum_returns_min = (min_prices_return / close[loc] - 1) * side
        cum_returns_max = (max_prices_return / close[loc] - 1) * side

        sl_date = cum_returns_min[cum_returns_min < stop_loss[loc]].index.min()
        pt_date = cum_returns_max[cum_returns_max > profit_taking[loc]].index.min()

        vert_barrier_time = events_['t1'].loc[loc]

        out.loc[loc, 'touched_date'] = vert_barrier_time
        out.loc[loc, 'profit_perc_vol'] = 0.0
        is_sl_date_nan = pd.isna(sl_date)
        is_pt_date_nan = pd.isna(pt_date)

        if is_sl_date_nan and is_pt_date_nan:
            continue

        if is_sl_date_nan and not is_pt_date_nan and pt_date <= vert_barrier_time:
            out.loc[loc, 'touched_date'] = pt_date
            out.loc[loc, 'profit_perc_vol'] = profit_taking[loc]
            continue

        if is_pt_date_nan and not is_sl_date_nan and sl_date <= vert_barrier_time:
            out.loc[loc, 'touched_date'] = sl_date
            out.loc[loc, 'profit_perc_vol'] = stop_loss[loc]
            continue

        if not is_sl_date_nan and not is_pt_date_nan:
            if sl_date < pt_date and sl_date <= vert_barrier_time:
                out.loc[loc, 'touched_date'] = sl_date
                out.loc[loc, 'profit_perc_vol'] = stop_loss[loc]
            elif pt_date < sl_date and pt_date <= vert_barrier_time:
                out.loc[loc, 'touched_date'] = pt_date
                out.loc[loc, 'profit_perc_vol'] = profit_taking[loc]

    return out


def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
    nearest_index = close.index.searchsorted(t_events + timedelta)
    nearest_index = nearest_index[nearest_index < close.shape[0]]
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]
    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers


def get_events(bars, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False,
               side_prediction=None):
    target = target.loc[t_events]
    target = target[target > min_ret]
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.loc[target.index]
        pt_sl_ = pt_sl[:2]
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      bars=bars,
                                      events=events,
                                      pt_sl=pt_sl_)

    for ind in events.index:
        events.loc[ind, 't1'] = first_touch_dates.loc[ind, 'touched_date']
        events.loc[ind, 'profit_perc_vol'] = first_touch_dates.loc[ind, 'profit_perc_vol']
    if side_prediction is None:
        events = events.drop('side', axis=1)
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]
    return events


def meta_labeling(triple_barrier_events):
    """
    
    If the side is present in the events, all negative bin -1 are set to 0, 
    only positive returns remains.
    
    """

    out_df = pd.DataFrame(index=triple_barrier_events.index)
    out_df['ret'] = triple_barrier_events['profit_perc_vol']
    out_df['side'] = triple_barrier_events['side']

    # o side are not touched barriers
    out_df['ret'] = out_df.apply(lambda row: 0 if row['side'] == 0 else row['ret'], axis=1)

    # sl, pt and not touched
    out_df['bin'] = out_df['ret'].apply(lambda r: 1 if r > 0 else (-1 if r < 0 else 0))
    
    # set what is not touched or negative to 0
    out_df.loc[out_df['ret'] <= 0, 'bin'] = 0

    out_df['side'] = triple_barrier_events['side']

    return out_df


def drop_labels(events, min_pct=0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events
