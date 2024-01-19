from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.base_bars import BaseBars

class StandardBars(BaseBars):
    def __init__(self, metric: str,
                 threshold: int = 50000,
                 batch_size: int = 20000000,
                 symbol: str = None,
                 file_format: str = '.csv'):

        BaseBars.__init__(self, metric, batch_size, symbol, file_format)
        self.threshold = threshold

    def _reset_cache(self):
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0,
                               'cum_dollar_value': 0,
                               'cum_volume': 0,
                               'cum_buy_volume': 0,
                               'cum_buyer_market_maker': 0,
                               'cum_buyer_market_maker_volume': 0}


    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        list_bars = []
        timestamps = data.index.values
        prices = data.price.values
        volumes = data.quantity.values
        buyerMarketMakers = data.buyerMarketMaker.values

        for timestamp, price, volume, buyerMarketMaker in zip(timestamps,prices,volumes, buyerMarketMakers):
            # Set variables
            date_time = timestamp
            self.tick_num += 1
            price = np.float64(price)
            volume = np.float64(volume)
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            self.high_price, self.low_price = self._update_high_low(price)

            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume

            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            if buyerMarketMaker == 1:
                self.cum_statistics['cum_buyer_market_maker'] += 1
                self.cum_statistics['cum_buyer_market_maker_volume'] += volume

            if self.cum_statistics[self.metric] >= self.threshold:
                self._create_bars(date_time,
                                  price,
                                  self.high_price,
                                  self.low_price,
                                  list_bars)
                self._reset_cache()
        return list_bars

def dollar_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
               symbol: str = None,
               file_format: str = '.csv',
               threshold: float = 1000000,
               batch_size: int = 1000000,
               verbose: bool = True,
               to_csv: bool = False,
               output_path: Optional[str] = None):

    bars = StandardBars(metric='cum_dollar_value',
                        threshold=threshold,
                        batch_size=batch_size,
                        symbol=symbol,
                        file_format=file_format)

    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                 verbose=verbose,
                                 to_csv=to_csv,
                                 output_path=output_path)

    dollar_bars.set_index('date_time', inplace=True)
    # second removal of duplicates
    dollar_bars = dollar_bars[~dollar_bars.index.duplicated(keep='first')]

    return dollar_bars

def volume_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
               symbol: str = None,
               file_format: str = '.csv',
               threshold: float = 10000,
               batch_size: int = 1000000,
               verbose: bool = True,
               to_csv: bool = False,
               output_path: Optional[str] = None):
    bars = StandardBars(metric='cum_volume',
                        threshold=threshold,
                        batch_size=batch_size,
                        symbol = symbol,
                        file_format = file_format)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars

def tick_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
             symbol: str = None,
             file_format: str = '.csv',
             threshold: float = 600,
             batch_size: int = 1000000,
             verbose: bool = True,
             to_csv: bool = False,
             output_path: Optional[str] = None):
    bars = StandardBars(metric='cum_ticks',
                        threshold=threshold,
                        batch_size=batch_size,
                        symbol = symbol,
                        file_format = file_format)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars