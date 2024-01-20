import argparse
from tardis_dev import datasets
import functools
import itertools
import pandas as pd
import numpy as np
import gzip
import os
import gc
import time
import multiprocessing
from joblib import delayed, Parallel
from datetime import datetime, timedelta
import random

MAX_SYMBOLS_PER_RUN = 6
API_KEY = "TD.7nY1VBeQdUxxMyMK.BDpk0GcNRMraF7E.Xnov8iHJGsDqKgF.ELjyVH474rltzA7.kuy6pcsxCVRI2N-.X7yr"
DATA_DIR = os.path.join(os.getcwd(), "datasets")
SAVE_DIR = os.path.join(os.getcwd(), 'bars')
BAR_INTERVAL_MINS = 7
MAX_PROCESSES = 2


def get_date_range_by_month(start_date_str, end_date_str):
    r = pd.date_range(start=start_date_str, end=end_date_str)
    r = [str(s) for s in r.strftime("%Y-%m-%d")]

    partitioned_dates = []
    current_month = None
    current_partition = []

    # Iterate through date strings and partition them by month
    for date_str in r:
        year, month, _ = date_str.split("-")
        month_key = f"{year}-{month}"

        if month_key != current_month:
            if current_partition:
                partitioned_dates.append(current_partition)
            current_partition = []
            current_month = month_key

        current_partition.append(date_str)

    # Append the last partition
    if current_partition:
        partitioned_dates.append(current_partition)

    return partitioned_dates


def parse_symbols(symbols_str):
    return symbols_str.split(',')


def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_permutations_to_download(symbols, start_date_str, end_date_str):
    r = get_date_range_by_month(start_date_str, end_date_str)
    s = chunk_list(symbols, MAX_SYMBOLS_PER_RUN)
    return list(itertools.product(r, s))


def get_all_date_symbol_combos(symbols, start_date_str, end_date_str):
    r = get_date_range_by_month(start_date_str, end_date_str)
    s = chunk_list(symbols, MAX_SYMBOLS_PER_RUN)
    return list(itertools.product(flatten_list_of_lists(r), flatten_list_of_lists(s)))


def create_bars_wrapper(arg):
    date, symbol = arg
    create_bars(date, symbol, BAR_INTERVAL_MINS, SAVE_DIR)


def create_bars(date, symbol, bar_interval_mins, save_dir):
    save_path = os.path.join(save_dir, '_'.join([symbol, date, str(bar_interval_mins)])) + '.csv'

    if os.path.exists(save_path):
        return

    def gz2df(path, dtype):
        columns = [d[0] for d in dtype]
        with gzip.open(path, 'rt') as file:
            data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype=dtype, names=True)
        df = pd.DataFrame(data)
        df.columns = columns
        return df

    DTYPES = {'quotes':
                [('exchange', 'U20'),
                ('symbol', 'U20'),
                ('timestamp', np.int64),
                ('local_timestamp', np.int64),
                ('ask_amount', np.float64),
                ('ask_price', np.float64),
                ('bid_price', np.float64),
                ('bid_amount', np.float64),
                ],
            'trades':
                [('exchange', 'U20'),
                ('symbol', 'U20'),
                ('timestamp', np.int64),
                ('local_timestamp', np.int64),
                ('id', np.int64),
                ('side', 'U8'),
                ('price', np.float64),
                ('amount', np.float64),
                ],
            'deriv':
                [('exchange', 'U20'),
                ('symbol', 'U20'),
                ('timestamp', np.int64),
                ('local_timestamp', np.int64),
                ('funding_timestamp', np.int64),
                ('funding_rate', np.float64),
                ('predicted_funding_rate', np.float64),
                ('open_interest', np.float64),
                ('last_price', np.float64),
                ('index_price', np.float64),
                ('mark_price', np.float64),
                ],
            }

    trade_path = None
    quote_path = None
    deriv_path = None
    for p in filter(lambda x: date in x and symbol.upper() in x, os.listdir(DATA_DIR)):
        if 'quotes' in p:
            quote_path = os.path.join(DATA_DIR, p)
        elif 'trades' in p:
            trade_path = os.path.join(DATA_DIR, p)
        elif 'deriv' in p:
            deriv_path = os.path.join(DATA_DIR, p)
    try:
        needed_cols = ['timestamp', 'ask_amount', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount', 'open_interest', 'funding_rate', 'side', 'amount', 'price']
        q_df = gz2df(quote_path, DTYPES['quotes'])
        q_df.drop(columns=list(filter(lambda x: x not in needed_cols, q_df.columns)), inplace=True)
        t_df = gz2df(trade_path, DTYPES['trades'])
        t_df.drop(columns=list(filter(lambda x: x not in needed_cols, t_df.columns)), inplace=True)
    except TypeError:
        print(f'Error: {date} - {symbol}')
        return
    q_df['wm'] = (q_df.ask_amount * q_df.bid_price + q_df.bid_amount * q_df.ask_price) / (q_df.bid_amount + q_df.ask_amount)
    t_df['side'] = t_df['side'].map({"buy": 1, "sell": -1})
    q_df['timestamp'] = pd.to_datetime(q_df['timestamp'], unit='us')
    t_df['timestamp'] = pd.to_datetime(t_df['timestamp'], unit='us')
    q_df.set_index('timestamp', inplace=True)
    t_df.set_index('timestamp', inplace=True)

    df = q_df.join(t_df)
    del q_df
    del t_df
    gc.collect()

    try:
        d_df = gz2df(deriv_path, DTYPES['deriv'])
        d_df.drop(columns=list(filter(lambda x: x not in needed_cols, d_df.columns)), inplace=True)
    except TypeError:
        print(f'Error: {date} - {symbol}')
        return
    d_df['timestamp'] = pd.to_datetime(d_df['timestamp'], unit='us')
    d_df.set_index('timestamp', inplace=True)
    df = df.join(d_df)
    del d_df
    gc.collect()

    earliest_timestamp = df.index.min()
    nearest_midnight = earliest_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    bar_interval = pd.Timedelta(minutes=bar_interval_mins)
    bar_boundaries = pd.date_range(start=nearest_midnight, end=df.index.max(), freq=bar_interval)
    bars = pd.DataFrame(index=bar_boundaries)

    merged_data = pd.merge(bars, df, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    del bars
    del df
    gc.collect()

    merged_data['O'] = merged_data['wm']
    merged_data['H'] = merged_data['wm']
    merged_data['L'] = merged_data['wm']
    merged_data['C'] = merged_data['wm']
    merged_data['V'] = merged_data['amount']
    merged_data['TWAP'] = merged_data['wm']
    merged_data['OI'] = merged_data['open_interest']
    merged_data['FR'] = merged_data['funding_rate']
    merged_data['HT'] = merged_data['wm']
    merged_data['LT'] = merged_data['wm']
    merged_data['SV'] = merged_data['side'] * merged_data['amount']
    merged_data['PV'] = merged_data['amount'] * merged_data['price']

    df = merged_data.resample(str(bar_interval_mins) + 'T').apply({
        'O': 'first',
        'H': 'max',
        'L': 'min',
        'C': 'last',
        'V': 'sum',
        'TWAP': 'mean',
        'FR': 'last',
        'OI': 'last',
        'HT': lambda x: x.idxmax(),
        'LT': lambda x: x.idxmin(),
        'SV': 'sum',
        'PV': 'sum',
    })

    del merged_data
    gc.collect()

    df.to_csv(save_path, header=0)
    del df
    gc.collect()

    os.remove(trade_path)
    os.remove(quote_path)
    os.remove(deriv_path)


def main():
    parser = argparse.ArgumentParser(description="getbars")

    parser.add_argument("--from_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--symbols", type=parse_symbols, help="Comma-separated list of financial symbols or identifiers")

    args = parser.parse_args()

    from_date = args.from_date
    to_date = args.to_date
    symbols = args.symbols

    # Now you can use the start_date, end_date, and symbol in your script
    print(f"Start Date: {from_date}")
    print(f"End Date: {to_date}")
    print(f"Symbols: {symbols}")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    download_func = functools.partial(
        datasets.download,
        exchange='binance-futures',
        data_types=['trades', 'quotes', 'derivative_ticker'],
        api_key=API_KEY)

    random.shuffle(symbols)
    for p in get_permutations_to_download(symbols, from_date, to_date):
        print(p[0], p[1])
        this_from_date = p[0][0]
        this_to_date = p[0][-1]
        this_to_date_shifted_dt = datetime.strptime(this_to_date, '%Y-%m-%d')
        this_to_date_shifted_dt += timedelta(days=1)
        this_to_date_shifted = this_to_date_shifted_dt.strftime('%Y-%m-%d')

        this_symbols = p[1]
        print(f'Downloading bars for {this_symbols} from {this_from_date} to {this_to_date_shifted}')
        start_time = time.time()
        download_func(from_date=this_from_date, to_date=this_to_date_shifted, symbols=this_symbols)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Done, took {elapsed_time:.6f} seconds')

        arg_list = get_all_date_symbol_combos(this_symbols, this_from_date, this_to_date)
        # pool = multiprocessing.Pool(processes=MAX_PROCESSES)
        # pool.map(create_bars_wrapper, arg_list)
        # pool.close()
        # pool.join()
        Parallel(n_jobs=MAX_PROCESSES)(delayed(create_bars_wrapper)(arg) for arg in arg_list)

if __name__ == "__main__":
    main()