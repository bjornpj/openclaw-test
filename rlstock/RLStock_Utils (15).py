import os
import pickle
import pandas as pd
import ta  # Technical analysis library

#######################################################
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import ta

import torch as T

import io
import PIL
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
#from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import threading
from tqdm import tqdm
from numba import njit
#######################################################

MAX_ACCOUNT_BALANCE = 20000
MAX_NUM_SHARES = 4000
MAX_SHARE_PRICE = 6
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

# TupleBuffer Class
import numpy as np
from collections import defaultdict

class TupleBuffer:
    def __init__(self):
        # Store data as a dictionary of lists, keyed by (tick, step)
        self.buffer = []
        self.indices = defaultdict(list)

    def __len__(self):
        return sum(len(v) for v in self.indices.values())

    def add_tuple(self, tups):
        # Assumes tups = (tick, step, ...) and we want to store from index 2 onward
        key = (tups[0], tups[1])
        self.indices[key].append(tups[2:])

    def get_indices(self, tups):
        # Fast dictionary lookup
        return self.indices.get((tups[0], tups[1]), [])

# DataEnv Class
class DataEnv:
    def __init__(self, save_tuple_buffer=True, path="data/", fname="_last_10_years", ndays=16):
        self.save_tuple_buffer = save_tuple_buffer
        self.path = path
        self.fname = fname
        self.tbuf_train, self.tbuf_test = TupleBuffer(), TupleBuffer()
        self.date_ranges = {}
        self.terminate = False  # Initialize terminate flag
        self.ndays = ndays

    def process(self):
        data_directory = self.path
        train_directory = os.path.join(data_directory, 'train/')
        test_directory = os.path.join(data_directory, 'test/')
        os.makedirs(train_directory, exist_ok=True)
        os.makedirs(test_directory, exist_ok=True)
    
        all_files = os.listdir(data_directory)
    
        # Only consider files that match the expected pattern
        valid_files = [f for f in all_files if f.endswith(str(self.fname+".csv"))]
        tickers = [fname.split("_")[0] for fname in valid_files]
    
        # Create for each time step a record of previous 15 timesteps
        xdecimals = 4
        xClose = 'Adj Close'
        cnt = 1
        self.date_ranges = {}

#        tickers = ['1COV.DE', 'AAL', 'AAPL']
    
        for ticker in tickers:
            # Convert data to float16 while reading it
            df = pd.read_csv(f"{data_directory}\{ticker}_{self.fname}.csv", parse_dates=["Date"]).set_index("Date")
#            df = pd.read_csv(os.path.join(data_directory, f"{ticker}_{self.fname}.csv"), parse_dates=['Date']).set_index('Date').astype('float16')
    
            # Check if # rows > 1050. Train + Test = 700 + 350 = 1050
            if df.shape[0] > 1050:
                if df.Close.max() > 750:
                    # Scale high priced tickers to normal range
                    df = df / 100
                    
                adj_factor = df['Adj Close'] / df['Close']
    
                # Calculate Bollinger Bands using the 'ta' library
                bollinger_indicator = ta.volatility.BollingerBands(close=df[xClose], window=14)
                df['BMA'] = bollinger_indicator.bollinger_mavg().round(xdecimals)
                df['BUL'] = bollinger_indicator.bollinger_hband().round(xdecimals)
                df['BLL'] = bollinger_indicator.bollinger_lband().round(xdecimals)
        
                # Calculate Keltner Channel using the 'ta' library
                keltner_indicator = ta.volatility.KeltnerChannel(close=df[xClose], high=df['High'] * adj_factor, low=df['Low'] * adj_factor, window=20)
                df['KMA'] = keltner_indicator.keltner_channel_mband().round(xdecimals)
                df['KUL'] = keltner_indicator.keltner_channel_hband().round(xdecimals)
                df['KLL'] = keltner_indicator.keltner_channel_lband().round(xdecimals)
        
                # MACD (Moving Average Convergence Divergence)
                macd_indicator = ta.trend.MACD(close=df[xClose], window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd_indicator.macd().round(xdecimals)
                df['MACD_signal'] = macd_indicator.macd_signal().round(xdecimals)
                df['MACD_diff'] = macd_indicator.macd_diff().round(xdecimals)  # Histogram
        
                # EMA (Exponential Moving Averages)
                df['EMA_12'] = ta.trend.sma_indicator(close=df[xClose], window=9).round(xdecimals)
                df['EMA_26'] = ta.trend.sma_indicator(close=df[xClose], window=45).round(xdecimals)
#                df['EMA_12'] = ta.trend.ema_indicator(close=df[xClose], window=12).round(xdecimals)
#                df['EMA_26'] = ta.trend.ema_indicator(close=df[xClose], window=26).round(xdecimals)
        
                # Calculate RSI using the 'ta' library
                rsi_indicator = ta.momentum.RSIIndicator(close=df[xClose], window=14)
        
                df['RSI'] = rsi_indicator.rsi().round(xdecimals) / 100
                
                # Calculate ADX using the 'ta' library
                adx_indicator = ta.trend.ADXIndicator(close=df[xClose], high=df['High'] * adj_factor, low=df['Low'] * adj_factor, window=14)
                df['ADI'] = adx_indicator.adx().round(xdecimals) / 100
                df['NDI'] = adx_indicator.adx_neg().round(xdecimals) / 100
                df['PDI'] = adx_indicator.adx_pos().round(xdecimals) / 100
                
                # Convert to Percentage Change
                dfc = df.copy()
                for col in ['Open','Close','Adj Close','High','Low','BMA','BUL','BLL','KMA','KUL','KLL','MACD','MACD_signal','MACD_diff','EMA_12','EMA_26']:
                    dfc[col] = (df[col].astype('float64').pct_change()) + 1
                
                # Delete first 30 rows due to moving avg calcs creates zeros
                df = df.iloc[50:]
                dfc = dfc.iloc[50:]
#                df = df.iloc[31:]
#                dfc = dfc.iloc[31:]
                
                # Split data into train and test sets
                train_df, test_df = self.split_data(df, split_ratio=0.7)
                train_dfc, test_dfc = self.split_data(dfc, split_ratio=0.7)
                
                # Extract the minimum and maximum dates from the index
                train_row_count, test_row_count = train_df.shape[0], test_df.shape[0]
                # Store the date range in the dictionary
                self.date_ranges[ticker] = (train_row_count, test_row_count)
                
                # Create for each Time Step (TS) a record of previous 15 TS's
                tstep = self.ndays - 1
#z200                tstep = 15
                print(f'{cnt} train & test created for {ticker}')
                cnt += 1
                for i in range(tstep+1, train_row_count - 1):
                    self.tbuf_train.add_tuple(self.proc_obs_large(train_df.iloc, train_dfc.iloc, tstep, i , ticker))
                
                for i in range(tstep+1, test_row_count - 1):
                    self.tbuf_test.add_tuple(self.proc_obs_large(test_df.iloc, test_dfc.iloc, tstep, i , ticker))
                
    def proc_obs_large(self, dta, dtac, twindow, current_step, tick):
        # Get the stock data points for the last 5 days and scale to between 0-1
        xto = current_step
        xs0 = twindow
        XS1 = XS0 = 1
        repos = 1
    
        curcls = dta[xto]['Adj Close']
        nxtcls = dta[xto+1]['Adj Close']
        curbma = dta[xto]['BMA']
        curbul = dta[xto]['BUL']
        curbll = dta[xto]['BLL']
    
        data_slice = slice(xto - xs0, xto + 1)
    
        cal_data = dta[data_slice]
        cal_close = cal_data['Adj Close']
    
        calbma = (cal_data['BMA'] - cal_close) / cal_close
        calbul = (cal_data['BUL'] - cal_close) / cal_close
        calbll = (cal_data['BLL'] - cal_close) / cal_close
        calkma = (cal_data['KMA'] - cal_close) / cal_close
        calkul = (cal_data['KUL'] - cal_close) / cal_close
        calkll = (cal_data['KLL'] - cal_close) / cal_close
        calopn = (cal_data['Open'] - cal_close) / cal_close
        callow = (cal_data['Low'] - cal_close) / cal_close
        calhgh = (cal_data['High'] - cal_close) / cal_close
        calem12 = (cal_data['EMA_12'] - cal_close) / cal_close
        calem26 = (cal_data['EMA_26'] - cal_close) / cal_close


        cal_close = cal_data['Adj Close'].to_numpy()
        zcls = cal_close / cal_close[0]
#1006        zcls = dtac[data_slice]['Adj Close'].cumprod().values
    
        zbma = zcls * (1 + calbma).values
        zbul = zcls * (1 + calbul).values
        zbll = zcls * (1 + calbll).values
        zkma = zcls * (1 + calkma).values
        zkul = zcls * (1 + calkul).values
        zkll = zcls * (1 + calkll).values
        zopn = zcls * (1 + calopn).values
        zlow = zcls * (1 + callow).values
        zhgh = zcls * (1 + calhgh).values
        zem12 = zcls * (1 + calem12).values
        zem26 = zcls * (1 + calem26).values
    
        zrsi = dtac[data_slice]['RSI'].values
        zadi = dtac[data_slice]['ADI'].values
        zpdi = dtac[data_slice]['PDI'].values
        zndi = dtac[data_slice]['NDI'].values
    
        return tick, current_step, curcls, nxtcls, curbma, curbul, curbll, zbma, zbul, zbll, zkma, zkul, zkll, zcls, zrsi, zadi, zpdi, zndi, zhgh, zlow, zopn, zem12, zem26  
    
    def split_data(self, df, split_ratio=0.8):
        # Split the data into training and testing sets based on the given split ratio
        split_index = int(len(df) * split_ratio)
        return df[:split_index], df[split_index:]

    def process_save(self, fname):
        print(f"{self.path}\\tuplebuffer_train_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\tuplebuffer_train_{fname}_{self.ndays}d.pickle", 'wb') as f:
            pickle.dump(self.tbuf_train, f)
    
        print(f"{self.path}\\tuplebuffer_test_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\tuplebuffer_test_{fname}_{self.ndays}d.pickle", 'wb') as f:
            pickle.dump(self.tbuf_test, f)
    
        print(f"{self.path}\\date_ranges_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\date_ranges_{fname}_{self.ndays}d.pickle", 'wb') as f:
            pickle.dump(self.date_ranges, f)
    
    def process_load(self, fname):
        print(f"loading {self.path}\\tuplebuffer_train_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\tuplebuffer_train_{fname}_{self.ndays}d.pickle", 'rb') as f:
            self.tbuf_train = pickle.load(f)
    
        print(f"loading {self.path}\\tuplebuffer_test_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\tuplebuffer_test_{fname}_{self.ndays}d.pickle", 'rb') as f:
            self.tbuf_test = pickle.load(f)
    
        print(f"loading {self.path}\\date_ranges_{fname}_{self.ndays}d.pickle")
        with open(f"{self.path}\\date_ranges_{fname}_{self.ndays}d.pickle", 'rb') as f:
            self.date_ranges = pickle.load(f)

    
#@njit(cache=True, fastmath=True, nogil=True)
@njit
def compute_observation(zcls, zopn, zlow, zhgh, zbma, zbul, zbll, zem12, zem26,
                        zpdi, zndi, zadi, zrsi, calibrate, calibrate_aux,
                        buysell, ndays):
    calibrated_features = np.empty((9, ndays), dtype=np.float32)
    calibrated_aux = np.empty((4, ndays), dtype=np.float32)
    
    for i in range(ndays):
        calibrated_features[0, i] = zcls[i] * calibrate#calibrate[0]
        calibrated_features[1, i] = zopn[i] * calibrate#calibrate[1]
        calibrated_features[2, i] = zlow[i] * calibrate#calibrate[2]
        calibrated_features[3, i] = zhgh[i] * calibrate#calibrate[3]
        calibrated_features[4, i] = zbma[i] * calibrate#calibrate[4]
        calibrated_features[5, i] = zbul[i] * calibrate#calibrate[5]
        calibrated_features[6, i] = zbll[i] * calibrate#calibrate[6]
        calibrated_features[7, i] = zem12[i] * calibrate#calibrate[7]
        calibrated_features[8, i] = zem26[i] * calibrate#calibrate[8]

        calibrated_aux[0, i] = zpdi[i]# * calibrate_aux[0]
        calibrated_aux[1, i] = zndi[i]# * calibrate_aux[1]
        calibrated_aux[2, i] = zadi[i]# * calibrate_aux[2]
        calibrated_aux[3, i] = zrsi[i]# * calibrate_aux[3]

    ts_length = 10 * (ndays + 1)  # 10 features, each (ndays + 1) long with buysell prepended
    ts = np.empty((ts_length,), dtype=np.float32)
    idx = 0

    # Main features: BMA, Close, BUL, BLL, EMA12, EMA26
    for i in [4, 0, 5, 6, 7, 8]:
        ts[idx] = buysell
        idx += 1
        for j in range(ndays):
            ts[idx] = calibrated_features[i, j]
            idx += 1

    # Aux features: RSI, PDI, NDI, ADI
    for i in [3, 0, 1, 2]:
        ts[idx] = buysell
        idx += 1
        for j in range(ndays):
            ts[idx] = calibrated_aux[i, j]
            idx += 1

    return ts.reshape(-1, ndays + 1, 1)

@njit(cache=True, fastmath=True)
def atr_proxy_percent_numba(zcls, zlow, zhgh, period=14, eps=1e-12):
    T = zcls.shape[0]
    a = 2.0 / (period + 1.0)
    one_ma = 1.0 - a
    # first TR at t=1
    cls_prev = zcls[0]
    high_t = zhgh[1]
    low_t  = zlow[1]
    tr = max(abs(high_t - low_t), abs(high_t - cls_prev), abs(low_t - cls_prev))
    ema = tr
    # rest
    for t in range(2, T):
        cls_prev = zcls[t-1]
        high_t = zhgh[t]
        low_t  = zlow[t]
        tr = max(abs(high_t - low_t), abs(high_t - cls_prev), abs(low_t - cls_prev))
        ema = one_ma * ema + a * tr
    return ema if ema > eps else eps

def atr_proxy_from_percent_ohlc_np(zcls, zopn, zlow, zhgh, period=14, eps=1e-12):
    """
    z* are percent-change windows (length T, latest at index T-1).
    Returns an ATR-like volatility (in percent units) for the *latest* step.
    """
    zcls = np.asarray(zcls); zopn = np.asarray(zopn)
    zlow = np.asarray(zlow); zhgh = np.asarray(zhgh)
    assert zcls.ndim == 1, "pass a single window (shape [T])"
    T = zcls.shape[0]; assert T >= 2

    cls_prev = zcls[:-1]     # t-1
    high_t   = zhgh[1:]      # t
    low_t    = zlow[1:]      # t

    tr1 = np.abs(high_t - low_t)
    tr2 = np.abs(high_t - cls_prev)
    tr3 = np.abs(low_t  - cls_prev)
    tr  = np.maximum(tr1, np.maximum(tr2, tr3))  # [T-1]

    alpha = 2.0 / (period + 1.0)
    atr = tr[0]
    for x in tr[1:]:
        atr = (1.0 - alpha) * atr + alpha * x
    return float(max(atr, eps))

class StockTradingEnvBB(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataenv, sd, ndays=16):
        super(StockTradingEnvBB, self).__init__()
        
        np.random.seed(2021 + sd)
        random.seed(2021 + sd)
        self.ndays = ndays
        
        # Preallocations for Numba to fill:
        self._calib_features = np.empty((9, ndays),  dtype=np.float32)
        self._calib_aux      = np.empty((4, ndays),  dtype=np.float32)
        self._ts_flat        = np.empty(10*(ndays+1), dtype=np.float32)
        self._obs            = np.empty((19, ndays+1, 1), dtype=np.float32)
#1002        self._obs            = np.empty((18, ndays+1, 1), dtype=np.float32)
#1000        self._obs            = np.empty((16, ndays+1, 1), dtype=np.float32)

        self.tbuf_train, self.tbuf_test = dataenv.tbuf_train, dataenv.tbuf_test
        self.ticks = dataenv.ticks
        self.date_ranges = dataenv.date_ranges
        # Define observation space (example)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,ndays+1,1), dtype=np.float32)
#1002        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,ndays+1,1), dtype=np.float32)

        # Define action space (example)
        self.action_space = spaces.Discrete(7)
        self.btraining = True
        #delete self.rate  = np.array([1/3, 1/2, 1.], dtype=np.float32)
        #delete self.ratey = np.array([1/3, 1/3, 1/3], dtype=np.float32)
#################################################################################################################
        # NEW: fees/penalties & reward knobs used by _take_action
        self.FEE           = 0.0          # fixed fee per trade (set >0 if desired)
        self.FEE_RATE      = 0.0          # proportional fee (e.g., 0.0005 = 5 bps)
        self.TRADE_PENALTY = 0.012 #0.01
        self.ZSCALE        = 8.0
        
        # NEW: profit-convexity (hold winners) & round-trip friction (short holds)
        #delete self.PROFIT_CONVEXITY_K = 0.07# 0.06
        #delete self.PC_MAX_BONUS       = 0.003
        #delete self.RT_MIN_HOLD        = 12
        #delete self.RT_PENALTY_MULT    = 3.0
        
        # NEW: timing markers used by _take_action
        #delete self.position_open_step    = -1
        #delete self.entry_price_for_bonus = 0.0
        
        # NEW: optional debug counter
        self._dbg_blocked_buys = 0
        # optional default; you can also put this in __init__
        #delete self.COOLDOWN_STEPS = getattr(self, "COOLDOWN_STEPS", 2)

#################################################################################################################
        self.ATR_PERIOD   = int(getattr(self, "ATR_PERIOD", 14))
        self._atr_pct_cur = 1e-4  # cached per-step ATR in percent space
#################################################################################################################
        # toggle for new vs old state block
        self.USE_NEW_STATE_BLOCK = True   # set to True to use new 6-channel state
        
        # initial balance used for nw_log_ratio; set this when you reset
        self.init_balance = 10000.0        # or override in reset()
        
        # episode length hint (used for time_in_ep_frac)
        self.MAX_EP_STEPS = 252
        
        # EMA for trade activity
        self._trade_activity_ema = 0.0
        self.TRADE_EMA_ALPHA = 0.1
        self.CLOSE_PROFIT_BONUS = 0.004
    
    def _next_observation(self):
        tbuf = self.tbuf_train if self.btraining else self.tbuf_test
        data = tbuf.get_indices((self.rand_tick, self.current_step))[0]
        
        (curcls, nxtcls, curbma, curbul, curbll, 
         zbma, zbul, zbll, zkma, zkul, zkll, 
         zcls, zrsi, zadi, zpdi, zndi, 
         zhgh, zlow, zopn, zem12, zem26) = data
    
        # ---- prices for env state ----
        self.curpri = curcls
        self.nxtpri = nxtcls
    
        # ---- cache ATR proxy (percent domain) for this step ----
        try:
            self._atr_pct_cur = atr_proxy_percent_numba(
                zcls, zlow, zhgh, period=self.ATR_PERIOD
            )
        except Exception:
            self._atr_pct_cur = 1e-4  # conservative fallback
    
        # ---- your existing "buysell" scalar (keep as-is) ----
        buysell = 2 * self.balance / (self.balance + self._bought_qty * curcls) - 1
        # buysell = self.balance / (self.balance + self._bought_qty * curcls)
    
        # ---- base obs from your numba fn (10 TA channels expected) ----
        obs_base = compute_observation(
            zcls, zopn, zlow, zhgh, zbma, zbul, zbll, zem12, zem26,
            zpdi, zndi, zadi, zrsi,
            float(self.calibrate), 1.0,
            float(buysell), self.ndays
        )  # shape: [10, T1, 1]
        obs_base = obs_base.astype(np.float32)
        T1 = int(obs_base.shape[1])  # T+1
    
        # ============================================================
        # TOGGLE: old vs new state block
        #   - USE_NEW_STATE_BLOCK = False  -> old behavior
        #   - USE_NEW_STATE_BLOCK = True   -> new 6-channel Markov-ish state
        # ============================================================
        if getattr(self, "USE_NEW_STATE_BLOCK", False):
            # --------------------------------------------------------
            # NEW 6-CHANNEL STATE BLOCK
            # --------------------------------------------------------
            price = float(curcls)
            shares = float(self.shares_held[0])
            position_value = shares * price
    
            net_worth = self.balance + position_value
            denom_nw = max(net_worth, 1e-9)
    
            # 1) position & cash fractions [0,1]
            pos_frac  = position_value / denom_nw
            cash_frac = self.balance / denom_nw
    
            # 2) unrealized PnL (squashed)
            if getattr(self, "_bought_qty", 0) > 0 and getattr(self, "Last_buy_price", None):
                cost_basis = max(float(self.Last_buy_price), 1e-9)
                unreal_pnl_pct = (price - cost_basis) / cost_basis
            else:
                unreal_pnl_pct = 0.0
            unreal_pnl_norm = float(np.tanh(unreal_pnl_pct * 2.0))
    
            # 2b) NEW: trade-life features (since entry)
            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            
            if shares > 0.0 and entry_price > 0.0:
                perf_since_entry = (price - entry_price) / max(entry_price, 1e-9)
                perf_since_entry_norm = float(np.tanh(perf_since_entry * 3.0))
            
                bars_since_entry = float(getattr(self, "_hold_steps_since_buy", 0))
                bars_since_entry_norm = float(np.tanh(bars_since_entry / 64.0))
            else:
                perf_since_entry_norm = 0.0
                bars_since_entry_norm = 0.0
                perf_since_entry = 0.0  # <--- This was missing in the 'else' logic

            # 3) net-worth log ratio vs initial balance
            init_bal = float(getattr(self, "init_balance", 10000.0))
            nw_ratio = net_worth / max(init_bal, 1e-9)
            nw_log_ratio = float(np.clip(np.log(nw_ratio + 1e-8), -2.0, 2.0))
    
            # 4) episode progress [0,1]
            steps_in_ep = float(getattr(self, "since_last", 0))
            max_ep_steps = float(getattr(self, "MAX_EP_STEPS", 252))
            time_in_ep_frac = steps_in_ep / max(max_ep_steps, 1.0)
            time_in_ep_frac = float(np.clip(time_in_ep_frac, 0.0, 1.0))
    
            # 5) smoothed trading activity [0,1]
            trade_ema = float(getattr(self, "_trade_activity_ema", 0.0))
            trade_activity_ema = float(np.clip(trade_ema, 0.0, 1.0))

            # NEW: trade-life features (since entry)
#del            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            
#del            if shares > 0.0 and entry_price > 0.0:
#del                perf_since_entry = (price - entry_price) / max(entry_price, 1e-9)
#del                perf_since_entry_norm = float(np.tanh(perf_since_entry * 3.0))
            
#del                bars_since_entry = float(getattr(self, "_hold_steps_since_buy", 0))
#del                bars_since_entry_norm = float(np.tanh(bars_since_entry / 64.0))
#del            else:
#del                perf_since_entry_norm = 0.0
#del                bars_since_entry_norm = 0.0

            ###################################################################################
            # --- NEW: Bollinger-derived states (midline/upper/lower are scalar "current" values) ---
            bma = float(curbma)
            bul = float(curbul)
            bll = float(curbll)
            
            band_w = max(bul - bll, 1e-9)
            
            # position inside channel: 0=lower, 0.5=mid, 1=upper  -> normalize to [-1, +1]
            bb_pos = (price - bll) / band_w
            bb_pos_norm = float(np.clip(2.0 * bb_pos - 1.0, -1.0, 1.0))
            
            # bandwidth as regime proxy (compression vs expansion), softly squashed
            bb_width = band_w / max(abs(bma), 1e-9)
            bb_width_norm = float(np.tanh(bb_width * 8.0))
            ###################################################################################
    
#            # --- IMPROVED PORTFOLIO STATE ---
#            state_vec = np.array([
#                pos_frac,
#                cash_frac,
#                float(np.tanh(unreal_pnl_pct * 5.0)),       # Boosted gain (was 2.0)
#                nw_log_ratio,
#                bb_pos_norm,
#                bb_width_norm,
#                float(np.tanh(perf_since_entry * 5.0)),    # Boosted gain (was 3.0)
#                bars_since_entry_norm,
#                buysell
#            ], dtype=np.float32)
            
            # pack to (6, T1, 1)
            state_vec = np.array([
                pos_frac,
                cash_frac,
                unreal_pnl_norm,
                nw_log_ratio,
                bb_pos_norm,        # NEW (replaces time_in_ep_frac)
                bb_width_norm,       # NEW (replaces trade_activity_ema)
                perf_since_entry_norm,     # NEW
                bars_since_entry_norm,     # NEW
                buysell
            ], dtype=np.float32)  # (6,)
    
            state_block = np.tile(
                state_vec[:, None, None],   # (6, 1, 1)
                (1, T1, 1)                  # -> (6, T1, 1)
            ).astype(np.float32)
    
            obs = np.concatenate([obs_base, state_block], axis=0)  # [16, T1, 1]
    
        else:
            # --------------------------------------------------------
            # OLD 5-STATE + RAMP (your current behavior)
            # --------------------------------------------------------
            # lazy init state buffers (length T1 so time lines up)
            if not hasattr(self, "pos_hist") or getattr(self, "_state_buf_len", None) != T1:
                self.pos_hist  = deque([0.0]*T1, maxlen=T1)   # position fraction history [0..1]
                self.act_hist  = deque([0]*T1,   maxlen=T1)   # 0=hold, 1=sell, 2=buy
                self._state_buf_len = T1
    
            pos_arr = np.asarray(self.pos_hist, dtype=np.float32)             # [T1]
            in_pos  = (pos_arr > 0.0).astype(np.float32)                      # [T1]
    
            # one-hot last actions per bar
            act_idx = np.asarray(self.act_hist, dtype=np.int64)               # [T1]
            act_idx = np.clip(act_idx, 0, 2)                                  # map 3->2 (buy max as buy)
            act_oh  = np.eye(3, dtype=np.float32)[act_idx]                    # [T1, 3]
            act_hold = act_oh[:, 0]
            act_sell = act_oh[:, 1]
            act_buy  = act_oh[:, 2]
    
            # stack state channels to [5, T1, 1]
            state_stack = np.stack(
                [in_pos, pos_arr, act_hold, act_sell, act_buy], axis=0
            ).astype(np.float32)[:, :, None]
    
            # positional ramp (oldest->newest = 0..1)
            ramp = np.linspace(0.0, 1.0, T1, dtype=np.float32)[None, :, None]  # [1, T1, 1]
    
            # final obs: 10 TA + 5 state + 1 ramp = 16
            obs = np.concatenate([obs_base, state_stack, ramp], axis=0)  # [16, T1, 1]
    
        # cache & return
        self._obs[...] = obs
        return obs
    
    """
    a) At sell phase and when trend is low-low, take buy action 
        with reward = -momentum * zscale. 
    b) At buy phase and when trend is high-high followed by at least nstep low-low, take sell action
        with reward = (net_worth - net_worth_before) / net_worth_before * zscale 

    T: ++---+-------++---+--------++++++---+----++++
    
    A: hhhhhhhhhhhhhhhhhhhhhhhhhhhBhhhhShhhhhhhhBhhh
    A: hhhhhhhhhhhhhhhhhhhhhhhhhhhbhhhhShhhhhhhhBhhh
    R: --+++-+++++++--+++-+++++++++++++++++-++++++++
    
    A: hhhhhhBhhhhhhhhhhhhhhhhhhhhBhhhhShhhhhhhhBhhh
    R: --+++-+++++++--+++-+++++++++++++++++-++++++++
    z801
                    T = Trend, R = Reward, A = Action
    """
######################################################################################################################################################
    def _take_action(self, action):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
    
        UPDATES INCLUDED:
          (A) FIX: reset entry anchors on ANY full flatten (profit OR loss)
          (B) NEW: realized PnL reward on EVERY sell (partial included), ATR-normalized
          (C) FIX #1 (best): excess-return reward vs Buy&Hold baseline (removes "flat is safest" pathology)
        """
    
        # ---------- fast locals ----------
        balance        = self.balance
        next_price     = self.nxtpri
        current_price  = self.curpri
        shares_held0   = self.shares_held[0]
    
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
        base_penalty   = float(getattr(self, "TRADE_PENALTY", 0.0006))
        zscale         = float(getattr(self, "ZSCALE", 8.0))  # kept for consistency if you use it elsewhere
    
        # simple volume fractions
        BUY_FRACS  = {1: 0.10, 2: 0.25, 3: 0.50}
        SELL_FRACS = {4: 0.10, 5: 0.25, 6: 0.50}
    
        # ---------- ATR-aware trade penalty ----------
        atr_pct_safe = max(self._atr_pct_cur, 1e-6)
        anneal = 0.5 + 0.5 * min(1.0, self.steps_since_reset / 64.0)
        atr_ref = 0.1
        beta = 0.25
        atr_weight = (atr_pct_safe / atr_ref) ** beta
        atr_weight = float(max(0.9, min(atr_weight, 1.2)))
        trade_penalty = base_penalty * anneal * atr_weight
    
        # ---------- invalid action penalty ----------
        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.01))
        INVALID_PEN_MAX  = float(getattr(self, "INVALID_PEN_MAX", 0.20))
        self._invalid_streak = getattr(self, "_invalid_streak", 0)
        invalid_flag = False
    
        # previous action (for flip penalty)
        prev_action = getattr(self, "_prev_action", 0)
    
        # price trace & net-worth before
        self.cprice.append(next_price)
        net_worth_before = balance + shares_held0 * current_price
    
        executed_buy  = False
        executed_sell = False
        full_close_profit = False   # did we fully close a profitable position?
    
        # close pnl buffer (used for close bonus)
        self._close_pnl_for_bonus = 0.0
    
        # realized pnl buffer for THIS STEP (partial sells included)
        self._realized_pnl_pct_step = 0.0
    
        # ---------- baseline setup (Buy & Hold) ----------
        # Make sure baseline is initialized even if reset() didn't do it yet.
        if (not getattr(self, "_baseline_is_init", False)) or (float(getattr(self, "_baseline_entry_price", 0.0)) <= 0.0):
            self._baseline_nw0 = float(net_worth_before)
            self._baseline_entry_price = float(current_price if current_price > 0.0 else next_price)
            self._baseline_is_init = True
    
        # cached baseline constants
        baseline_nw0    = float(getattr(self, "_baseline_nw0", net_worth_before))
        baseline_entry  = float(getattr(self, "_baseline_entry_price", current_price if current_price > 0.0 else next_price))
    
        # ---------- helpers ----------
        def _avg_cost():
            if getattr(self, "_bought_qty", 0) > 0:
                return self._bought_val / max(self._bought_qty, 1e-12)
            lbp = getattr(self, "Last_buy_price", None)
            return lbp if (lbp is not None and lbp > 0.0) else next_price
    
        def _apply_sale(qty):
            nonlocal balance, shares_held0, full_close_profit
    
            if qty <= 0:
                return False
    
            prev_pos = shares_held0
            qty = int(min(qty, prev_pos))
            if qty < 1:
                return False
    
            avg_cost_before = _avg_cost()
            notional        = qty * next_price
            fee_cash        = notional * fee_rate
    
            balance      += (notional - fee_cash)
            shares_held0 -= qty
    
            # proportional cost-basis adjustment
            cost_removed      = avg_cost_before * qty
            self._bought_qty  = max(0, int(getattr(self, "_bought_qty", 0)) - qty)
            self._bought_val  = max(0.0, float(getattr(self, "_bought_val", 0.0)) - cost_removed)
            self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else None
    
            # trackers
            self.total_shares_sold1.append(qty)
            self.total_sales_value1.append(notional)
            self._sold_qty = getattr(self, "_sold_qty", 0) + qty
            self._sold_val = getattr(self, "_sold_val", 0.0) + notional
    
            self.Last_sell_price = max(next_price, 0.0)
    
            # profit vs entry if available, else avg cost
            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            if entry_price > 0.0:
                self._close_pnl_for_bonus = (next_price - entry_price) / max(entry_price, 1e-9)
            else:
                self._close_pnl_for_bonus = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
    
            # realized pnl for THIS SELL STEP (weighted by fraction sold)
            pnl_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
            frac_sold = qty / max(prev_pos, 1e-9)
            self._realized_pnl_pct_step += float(pnl_pct) * float(frac_sold)
    
            # detect fully closed at profit
            if prev_pos > 0:
                profit_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
                if (qty == prev_pos) and (profit_pct > 0.0):
                    full_close_profit = True
    
            return True
    
        can_buy_one = balance >= (next_price * (1.0 + fee_rate))
    
        # -------------------------------------------------------
        # HOLD
        # -------------------------------------------------------
        if action == 0:
            pass
    
        # -------------------------------------------------------
        # BUY (1,2,3)
        # -------------------------------------------------------
        elif action in (1, 2, 3):
            if not can_buy_one:
                invalid_flag = True
                self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
            else:
                prev_pos0 = shares_held0
    
                frac = float(BUY_FRACS[action])
                cash_to_use = balance * frac
    
                denom = next_price * (1.0 + fee_rate)
                n_max = int(cash_to_use / max(denom, 1e-12))
    
                if n_max >= 1:
                    notional = n_max * next_price
                    fee_cash = notional * fee_rate
    
                    balance      -= (notional + fee_cash)
                    shares_held0 += n_max
    
                    # trackers
                    self.total_shares_bought1.append(n_max)
                    self.total_bought_value1.append(notional)
                    self._bought_qty = getattr(self, "_bought_qty", 0) + n_max
                    self._bought_val = getattr(self, "_bought_val", 0.0) + notional
    
                    self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else next_price
    
                    executed_buy = True
                    self._invalid_streak = 0
    
                    # mark fresh entry on 0 -> >0
                    if prev_pos0 == 0 and shares_held0 > 0:
                        self.position_open_step    = int(getattr(self, "steps_since_reset", 0))
                        self.entry_price_for_bonus = float(next_price)
                        self._hold_steps_since_buy = 0
                else:
                    invalid_flag = True
                    self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
    
        # -------------------------------------------------------
        # SELL (4,5,6)
        # -------------------------------------------------------
        elif action in (4, 5, 6):
            if shares_held0 <= 0:
                invalid_flag = True
            else:
                frac = float(SELL_FRACS[action])
                qty  = max(1, int(shares_held0 * frac))
                sold_ok = _apply_sale(qty)
                if sold_ok:
                    executed_sell = True
                    self._invalid_streak = 0
                else:
                    invalid_flag = True
    
        # -------------------------------------------------------
        # Write back state
        # -------------------------------------------------------
        self.balance = balance
        self.shares_held[0] = shares_held0
        self.net_worth = balance + shares_held0 * next_price
    
        # -------------------------------------------------------
        # REWARD SHAPING
        # -------------------------------------------------------
    
        # (C) FIX #1: EXCESS return vs baseline (Buy&Hold)
        profit_raw = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)
    
        baseline_nw_now = baseline_nw0 * (next_price / max(baseline_entry, 1e-9))
        baseline_raw = (baseline_nw_now - net_worth_before) / max(net_worth_before, 1e-9)
    
        profit_raw_excess = profit_raw - baseline_raw
    
        profit_reward = max(min(profit_raw_excess, 0.12), -0.12)
        r = profit_reward
    
        # 1b) ATR-normalized (risk-adjusted) component (use EXCESS too)
        ATR_FLOOR = float(getattr(self, "ATR_FLOOR", 0.003))
        ATR_CEIL  = float(getattr(self, "ATR_CEIL", 0.05))
    
        atr_eff = min(max(atr_pct_safe, ATR_FLOOR), ATR_CEIL)
        risk_adj = profit_raw_excess / atr_eff
        risk_adj = max(min(risk_adj, 4.0), -4.0)
    
        RISK_ADJ_WEIGHT = float(getattr(self, "RISK_ADJ_WEIGHT", 0.0015))
        r += RISK_ADJ_WEIGHT * risk_adj
    
        # (B) realized PnL reward on EVERY sell (partial included)
        REALIZED_SELL_W = float(getattr(self, "REALIZED_SELL_W", 0.004))
        if executed_sell and not invalid_flag:
            rp = float(getattr(self, "_realized_pnl_pct_step", 0.0))
            rp_adj = rp / max(atr_eff, 1e-6)
            rp_adj = max(min(rp_adj, 3.0), -3.0)
            r += REALIZED_SELL_W * float(np.tanh(rp_adj))
    
        # 2) Per-trade turnover cost (valid trades only)
        ACTION_LOAD = {
            0: 0.00,
            1: 1.10, 2: 1.25, 3: 1.40,
            4: 0.30, 5: 0.45, 6: 0.60
        }
        if (executed_buy or executed_sell) and not invalid_flag:
            load = ACTION_LOAD.get(action, 0.0)
            r -= trade_penalty * load
    
        # 3) Flip penalty – tame churn
        FLIP_PENALTY = float(getattr(self, "FLIP_PENALTY", 0.001))
        closing_trade = (
            (prev_action in (1, 2, 3) and action in (4, 5, 6)) or
            (prev_action in (4, 5, 6) and action in (1, 2, 3))
        )
        if (action != prev_action) and (executed_buy or executed_sell) and not invalid_flag:
            if not closing_trade:
                r -= FLIP_PENALTY
    
        # 4) Invalid action penalty (symmetric; prevents sell-spam)
        if invalid_flag:
            self._invalid_streak = min(self._invalid_streak + 1, 4)
            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
            r -= min(INVALID_POST_PEN * bump, INVALID_PEN_MAX)
            self._inv_ct = getattr(self, "_inv_ct", 0) + 1
        else:
            self._invalid_streak = 0
    
        # 5) Gentle risk term – penalize being very long in high ATR
        RISK_LAMBDA = float(getattr(self, "RISK_LAMBDA", 0.0005))
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac_step = float((shares_held0 * next_price) / denom_nw)
        r -= RISK_LAMBDA * pos_frac_step * atr_pct_safe
    
        # 5b) small bonus for holding winners (still uses absolute step profit sign)
        if (action == 0) and (shares_held0 > 0) and (profit_raw > 0) and not invalid_flag:
            r += 0.0002 * min(2.0, profit_raw * 6.0)
    
        # 6) Patience bonus in very low volatility for holding
        LOW_VOL_HOLD_ATR = float(getattr(self, "LOW_VOL_HOLD_ATR", 0.0025))
        HOLD_BONUS       = float(getattr(self, "HOLD_BONUS", 0.0001))
        if (action == 0) and not (executed_buy or executed_sell) and (atr_pct_safe < LOW_VOL_HOLD_ATR) and not invalid_flag:
            r += HOLD_BONUS
    
        # 7) Profit-sensitive de-risking bonus (partial trims in profit)
        if executed_sell and not invalid_flag and atr_pct_safe > 0.004:
            if pos_frac_step > 0.40:
                SELL_BONUS = float(getattr(self, "SELL_BONUS", 0.0003))
                r += SELL_BONUS
    
        # close profit bonus
        CLOSE_PROFIT_BONUS = float(getattr(self, "CLOSE_PROFIT_BONUS", 0.004))
        if full_close_profit and not invalid_flag:
            pnl = float(getattr(self, "_close_pnl_for_bonus", 0.0))
            if pnl > 0.0:
                r += CLOSE_PROFIT_BONUS * min(2.0, pnl * 10.0)
    
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # (A) FIX: reset entry markers on ANY full flatten (profit OR loss)
        if int(shares_held0) == 0:
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # store reward + prev action
        self.rewardt = r
        self._prev_action = action
    
        # track best NW
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
        # ---------- consistency snap for trackers ----------
        if getattr(self, "_bought_qty", 0) != self.shares_held[0]:
            self._bought_qty = int(self.shares_held[0])
            if self._bought_qty == 0:
                self._bought_val = 0.0
                self.Last_buy_price = None
            else:
                if self.Last_buy_price is not None and self.Last_buy_price > 0:
                    self._bought_val = self.Last_buy_price * self._bought_qty
    
        # ---------- histories ----------
        pos_frac = float((shares_held0 * next_price) / max(self.net_worth, 1e-9))
        if hasattr(self, "pos_hist"):
            self.pos_hist.append(pos_frac)
        if hasattr(self, "act_hist"):
            a = 0
            if executed_sell:
                a = 1
            elif executed_buy:
                a = 2
            self.act_hist.append(a)
    
        # ------------------------------------------------------------------
        # trade-lifecycle timers
        # ------------------------------------------------------------------
        shares_now = int(shares_held0)
    
        self._steps_since_trade = int(getattr(self, "_steps_since_trade", 0))
        if executed_buy or executed_sell:
            self._steps_since_trade = 0
        else:
            self._steps_since_trade += 1
    
        if executed_buy:
            self._last_trade_type = 1
        elif executed_sell:
            self._last_trade_type = -1
        else:
            self._last_trade_type = int(getattr(self, "_last_trade_type", 0))
    
        if shares_now > 0:
            if not executed_buy:
                self._hold_steps_since_buy = int(getattr(self, "_hold_steps_since_buy", 0)) + 1
        else:
            self._hold_steps_since_buy = 0
    
        self.steps_since_reset += 1
    
        # ---------- trade activity EMA ----------
        trade_flag = 1.0 if (executed_buy or executed_sell) else 0.0
        alpha = getattr(self, "TRADE_EMA_ALPHA", 0.1)
        prev_ema = getattr(self, "_trade_activity_ema", 0.0)
        self._trade_activity_ema = (1.0 - alpha) * prev_ema + alpha * trade_flag



    def _take_action_011826(self, action):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
    
        UPDATES INCLUDED:
          (A) FIX: reset entry anchors on ANY full flatten (profit OR loss)
          (B) Realized PnL reward on EVERY valid sell (partials included), ATR-normalized
          (C) Symmetric invalid-action penalty using _invalid_streak (stops sell-spam / buy-spam)
          (D) Tiny *gated* opportunity-cost when flat (helps escape HOLD/flat collapse early)
        """
    
        # ---------- fast locals (avoid repeated attribute lookups) ----------
        balance        = self.balance
        next_price     = self.nxtpri
        current_price  = self.curpri
        shares_held0   = self.shares_held[0]
    
        # hyperparams (cached once)
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
        base_penalty   = float(getattr(self, "TRADE_PENALTY", 0.0006))
        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.01))
        INVALID_PEN_MAX  = float(getattr(self, "INVALID_PEN_MAX", 0.20))
    
        FLIP_PENALTY   = float(getattr(self, "FLIP_PENALTY", 0.001))
        RISK_LAMBDA    = float(getattr(self, "RISK_LAMBDA", 0.0005))
        REALIZED_SELL_W = float(getattr(self, "REALIZED_SELL_W", 0.004))
        CLOSE_PROFIT_BONUS = float(getattr(self, "CLOSE_PROFIT_BONUS", 0.004))
    
        # ATR shaping
        atr_pct_safe   = self._atr_pct_cur if self._atr_pct_cur > 1e-6 else 1e-6
        ATR_FLOOR      = float(getattr(self, "ATR_FLOOR", 0.003))
        ATR_CEIL       = float(getattr(self, "ATR_CEIL", 0.05))
        RISK_ADJ_WEIGHT = float(getattr(self, "RISK_ADJ_WEIGHT", 0.0015))
    
        # annealed trade penalty
        steps_since_reset = int(getattr(self, "steps_since_reset", 0))
        anneal = 0.5 + 0.5 * (1.0 if steps_since_reset >= 64 else (steps_since_reset / 64.0))
        atr_ref = 0.1
        beta = 0.25
        atr_weight = (atr_pct_safe / atr_ref) ** beta
        if atr_weight < 0.9: atr_weight = 0.9
        elif atr_weight > 1.2: atr_weight = 1.2
        trade_penalty = base_penalty * anneal * atr_weight
    
        # action fractions (keep names; dict lookup cost is tiny vs rest)
        BUY_FRACS  = {1: 0.10, 2: 0.25, 3: 0.50}
        SELL_FRACS = {4: 0.10, 5: 0.25, 6: 0.50}
    
        # previous action (for flip penalty)
        prev_action = int(getattr(self, "_prev_action", 0))
    
        # invalid streak state
        self._invalid_streak = int(getattr(self, "_invalid_streak", 0))
        invalid_flag = False
    
        # price trace & net-worth before
        self.cprice.append(next_price)
        net_worth_before = balance + shares_held0 * current_price
    
        executed_buy  = False
        executed_sell = False
        full_close_profit = False
    
        # per-step buffers
        self._close_pnl_for_bonus = 0.0
        self._realized_pnl_pct_step = 0.0
    
        # ---------- helpers (keep for clarity; lightweight and used rarely) ----------
        def _avg_cost():
            bq = int(getattr(self, "_bought_qty", 0))
            if bq > 0:
                return float(getattr(self, "_bought_val", 0.0)) / max(bq, 1e-12)
            lbp = getattr(self, "Last_buy_price", None)
            return float(lbp) if (lbp is not None and lbp > 0.0) else float(next_price)
    
        def _apply_sale(qty):
            nonlocal balance, shares_held0, full_close_profit
    
            if qty <= 0:
                return False
    
            prev_pos = float(shares_held0)
            qty = int(min(qty, prev_pos))
            if qty < 1:
                return False
    
            avg_cost_before = _avg_cost()
            notional = qty * next_price
            fee_cash = notional * fee_rate
    
            balance      += (notional - fee_cash)
            shares_held0 -= qty
    
            # proportional cost-basis adjustment
            cost_removed = avg_cost_before * qty
            bq = int(getattr(self, "_bought_qty", 0)) - qty
            if bq < 0: bq = 0
            bv = float(getattr(self, "_bought_val", 0.0)) - cost_removed
            if bv < 0.0: bv = 0.0
            self._bought_qty = bq
            self._bought_val = bv
            self.Last_buy_price = (bv / bq) if bq > 0 else None
    
            # trackers
            self.total_shares_sold1.append(qty)
            self.total_sales_value1.append(notional)
            self._sold_qty = int(getattr(self, "_sold_qty", 0)) + qty
            self._sold_val = float(getattr(self, "_sold_val", 0.0)) + notional
            self.Last_sell_price = float(next_price) if next_price > 0 else 0.0
    
            # close pnl buffer (vs entry if available)
            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            if entry_price > 0.0:
                self._close_pnl_for_bonus = (next_price - entry_price) / max(entry_price, 1e-9)
            else:
                self._close_pnl_for_bonus = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
    
            # realized pnl for THIS sell step (weighted by fraction sold)
            pnl_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
            frac_sold = qty / max(prev_pos, 1e-9)
            self._realized_pnl_pct_step += float(pnl_pct) * float(frac_sold)
    
            # detect full close at profit (using avg_cost_before)
            if prev_pos > 0:
                profit_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
                if (qty == int(prev_pos)) and (profit_pct > 0.0):
                    full_close_profit = True
    
            return True
    
        # ---------- feasibility ----------
        can_buy_one = (balance >= (next_price * (1.0 + fee_rate)))
    
        # =======================================================
        # ACTION EXECUTION
        # =======================================================
        if action == 0:
            pass
    
        elif action in (1, 2, 3):
            if not can_buy_one:
                invalid_flag = True
                self._dbg_blocked_buys = int(getattr(self, "_dbg_blocked_buys", 0)) + 1
            else:
                prev_pos0 = shares_held0
    
                frac = float(BUY_FRACS[action])
                cash_to_use = balance * frac
    
                denom = next_price * (1.0 + fee_rate)
                n_max = int(cash_to_use / max(denom, 1e-12))
    
                if n_max >= 1:
                    notional = n_max * next_price
                    fee_cash = notional * fee_rate
    
                    balance      -= (notional + fee_cash)
                    shares_held0 += n_max
    
                    # trackers
                    self.total_shares_bought1.append(n_max)
                    self.total_bought_value1.append(notional)
                    self._bought_qty = int(getattr(self, "_bought_qty", 0)) + n_max
                    self._bought_val = float(getattr(self, "_bought_val", 0.0)) + notional
    
                    bq = int(self._bought_qty)
                    self.Last_buy_price = (float(self._bought_val) / bq) if bq > 0 else float(next_price)
    
                    executed_buy = True
    
                    # mark fresh entry on 0 -> >0
                    if prev_pos0 == 0 and shares_held0 > 0:
                        self.position_open_step    = int(steps_since_reset)
                        self.entry_price_for_bonus = float(next_price)
                        self._hold_steps_since_buy = 0
                else:
                    invalid_flag = True
                    self._dbg_blocked_buys = int(getattr(self, "_dbg_blocked_buys", 0)) + 1
    
        elif action in (4, 5, 6):
            if shares_held0 <= 0:
                invalid_flag = True
            else:
                frac = float(SELL_FRACS[action])
                qty  = max(1, int(shares_held0 * frac))
                if _apply_sale(qty):
                    executed_sell = True
                else:
                    invalid_flag = True
    
        # =======================================================
        # WRITE BACK STATE
        # =======================================================
        self.balance = balance
        self.shares_held[0] = shares_held0
        self.net_worth = balance + shares_held0 * next_price
    
        # =======================================================
        # REWARD SHAPING
        # =======================================================
        profit_raw = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)
        if profit_raw > 0.12: profit_raw = 0.12
        elif profit_raw < -0.12: profit_raw = -0.12
        r = profit_raw
    
        # ATR-normalized component
        atr_eff = atr_pct_safe
        if atr_eff < ATR_FLOOR: atr_eff = ATR_FLOOR
        elif atr_eff > ATR_CEIL: atr_eff = ATR_CEIL
    
        risk_adj = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)  # unclipped for risk_adj
        risk_adj = risk_adj / atr_eff
        if risk_adj > 4.0: risk_adj = 4.0
        elif risk_adj < -4.0: risk_adj = -4.0
        r += RISK_ADJ_WEIGHT * risk_adj
    
        # (B) realized pnl reward on every VALID sell (partials included)
        if executed_sell and not invalid_flag:
            rp = float(self._realized_pnl_pct_step)
            rp_adj = rp / max(atr_eff, 1e-6)
            if rp_adj > 3.0: rp_adj = 3.0
            elif rp_adj < -3.0: rp_adj = -3.0
            # tanh bound
            r += REALIZED_SELL_W * float(np.tanh(rp_adj))
    
        # per-trade turnover cost (valid trades only)
        if (executed_buy or executed_sell) and not invalid_flag:
            # small lookup table
            if action == 0: load = 0.0
            elif action == 1: load = 1.10
            elif action == 2: load = 1.25
            elif action == 3: load = 1.40
            elif action == 4: load = 0.30
            elif action == 5: load = 0.45
            else: load = 0.60
            r -= trade_penalty * load
    
        # flip penalty (avoid churn unless it's a direct close/open flip)
        if (executed_buy or executed_sell) and not invalid_flag and (action != prev_action):
            closing_trade = (
                (prev_action in (1, 2, 3) and action in (4, 5, 6)) or
                (prev_action in (4, 5, 6) and action in (1, 2, 3))
            )
            if not closing_trade:
                r -= FLIP_PENALTY
    
        # (C) invalid action penalty (symmetric) + streak
        if invalid_flag:
            self._invalid_streak = min(self._invalid_streak + 1, 4)
            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
            pen = INVALID_POST_PEN * bump
            if pen > INVALID_PEN_MAX: pen = INVALID_PEN_MAX
            r -= pen
            self._inv_ct = int(getattr(self, "_inv_ct", 0)) + 1
        else:
            self._invalid_streak = 0
    
        # gentle risk term – penalize being very long in high ATR
        denom_nw = self.net_worth if self.net_worth > 1e-9 else 1e-9
        pos_frac_step = float((shares_held0 * next_price) / denom_nw)
        r -= RISK_LAMBDA * pos_frac_step * atr_pct_safe
    
        # small bonus for holding winners (nudges HOLD vs early SELL)
        if (action == 0) and (shares_held0 > 0) and (profit_raw > 0) and (not invalid_flag):
            # profit_raw is clipped already
            r += 0.0002 * min(2.0, profit_raw * 6.0)
    
        # patience bonus in very low volatility for holding
        LOW_VOL_HOLD_ATR = float(getattr(self, "LOW_VOL_HOLD_ATR", 0.0025))
        HOLD_BONUS       = float(getattr(self, "HOLD_BONUS", 0.0001))
        if (action == 0) and (not executed_buy) and (not executed_sell) and (atr_pct_safe < LOW_VOL_HOLD_ATR) and (not invalid_flag):
            r += HOLD_BONUS
    
        # profit-sensitive de-risking bonus (partial trims in profit)
        if executed_sell and (not invalid_flag) and (atr_pct_safe > 0.004) and (pos_frac_step > 0.40):
            SELL_BONUS = float(getattr(self, "SELL_BONUS", 0.0003))
            r += SELL_BONUS
    
        # close-position bonus (only when fully closed profitably)
        if full_close_profit and (not invalid_flag):
            pnl = float(getattr(self, "_close_pnl_for_bonus", 0.0))
            if pnl > 0.0:
                r += CLOSE_PROFIT_BONUS * min(2.0, pnl * 10.0)
    
            # reset entry markers on full close
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # (A) FIX: reset entry markers on ANY full flatten (profit OR loss)
        if int(shares_held0) == 0:
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # (D) tiny opportunity-cost when flat (GATED; helps escape early HOLD/flat collapse)
        # IMPORTANT: keep tiny + temporary; do not bias long-run policy
        if (int(shares_held0) == 0) and (not invalid_flag) and (steps_since_reset < 300_000):
            r -= 0.00005
    
        # store reward + prev action
        self.rewardt = float(r)
        self._prev_action = int(action)
    
        # track best NW
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
        # ---------- consistency snap for trackers ----------
        if int(getattr(self, "_bought_qty", 0)) != int(self.shares_held[0]):
            self._bought_qty = int(self.shares_held[0])
            if self._bought_qty == 0:
                self._bought_val = 0.0
                self.Last_buy_price = None
            else:
                lbp = getattr(self, "Last_buy_price", None)
                if lbp is not None and lbp > 0:
                    self._bought_val = float(lbp) * float(self._bought_qty)
    
        # ---------- histories ----------
        if hasattr(self, "pos_hist"):
            self.pos_hist.append(pos_frac_step)
        if hasattr(self, "act_hist"):
            a = 0
            if executed_sell: a = 1
            elif executed_buy: a = 2
            self.act_hist.append(a)
    
        # ------------------------------------------------------------------
        # trade-lifecycle timers
        # ------------------------------------------------------------------
        shares_now = int(shares_held0)
    
        self._steps_since_trade = int(getattr(self, "_steps_since_trade", 0))
        if executed_buy or executed_sell:
            self._steps_since_trade = 0
        else:
            self._steps_since_trade += 1
    
        if executed_buy:
            self._last_trade_type = 1
        elif executed_sell:
            self._last_trade_type = -1
        else:
            self._last_trade_type = int(getattr(self, "_last_trade_type", 0))
    
        if shares_now > 0:
            if not executed_buy:  # don't increment on the entry bar
                self._hold_steps_since_buy = int(getattr(self, "_hold_steps_since_buy", 0)) + 1
        else:
            self._hold_steps_since_buy = 0
    
        self.steps_since_reset = steps_since_reset + 1
    
        # trade activity EMA
        trade_flag = 1.0 if (executed_buy or executed_sell) else 0.0
        alpha = float(getattr(self, "TRADE_EMA_ALPHA", 0.1))
        prev_ema = float(getattr(self, "_trade_activity_ema", 0.0))
        self._trade_activity_ema = (1.0 - alpha) * prev_ema + alpha * trade_flag




    def _take_action_old(self, action):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
    
        UPDATES INCLUDED (per your request):
          (A) FIX: reset entry anchors on ANY full flatten (profit OR loss)
          (B) NEW: realized PnL reward on EVERY sell (partial included), ATR-normalized
              - adds: self._realized_pnl_pct_step (weighted by fraction sold)
              - adds: REALIZED_SELL_W hyperparam (default 0.002)
        """
    
        # ---------- fast locals ----------
        balance        = self.balance
        next_price     = self.nxtpri
        current_price  = self.curpri
        shares_held0   = self.shares_held[0]
    
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
        # UPDATED: trade penalty scale default down 10x (was 0.006)
        base_penalty   = float(getattr(self, "TRADE_PENALTY", 0.0006))
        zscale         = float(getattr(self, "ZSCALE", 8.0))  # kept for consistency if you use it elsewhere
    
        # simple volume fractions
        BUY_FRACS  = {1: 0.10, 2: 0.25, 3: 0.50}
        SELL_FRACS = {4: 0.10, 5: 0.25, 6: 0.50}
    
        # ---------- ATR-aware trade penalty ----------
        atr_pct_safe = max(self._atr_pct_cur, 1e-6)
        anneal = 0.5 + 0.5 * min(1.0, self.steps_since_reset / 64.0)
        atr_ref = 0.1
        beta = 0.25
        # UPDATED: penalty increases with ATR (was inverted)
        atr_weight = (atr_pct_safe / atr_ref) ** beta
        atr_weight = float(max(0.9, min(atr_weight, 1.2)))
        trade_penalty = base_penalty * anneal * atr_weight
    
        # ---------- invalid action penalty ----------
        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.01))
#z801        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.05))
        INVALID_PEN_MAX  = float(getattr(self, "INVALID_PEN_MAX", 0.20))
        self._invalid_streak = getattr(self, "_invalid_streak", 0)
        invalid_flag = False
    
        # previous action (for flip penalty)
        prev_action = getattr(self, "_prev_action", 0)
    
        # price trace & net-worth before
        self.cprice.append(next_price)
        net_worth_before = balance + shares_held0 * current_price
    
        executed_buy  = False
        executed_sell = False
        full_close_profit = False   # did we fully close a profitable position?
    
        # NEW (b/c): per-trade close PnL buffer (used for close bonus)
        self._close_pnl_for_bonus = 0.0
    
        # NEW (B): realized pnl buffer for THIS STEP (partial sells included)
        self._realized_pnl_pct_step = 0.0
    
        #########################################################################################
        self.ATR_FLOOR = 0.003        # 0.3% effective minimum ATR
        self.ATR_CEIL  = 0.05         # 5% effective maximum ATR (optional)
        self.RISK_ADJ_WEIGHT = 0.0015
    
        # ---------- helpers ----------
        def _avg_cost():
            if getattr(self, "_bought_qty", 0) > 0:
                return self._bought_val / max(self._bought_qty, 1e-12)
            lbp = getattr(self, "Last_buy_price", None)
            return lbp if (lbp is not None and lbp > 0.0) else next_price
    
        def _apply_sale(qty):
            nonlocal balance, shares_held0, full_close_profit
    
            if qty <= 0:
                return False
    
            prev_pos = shares_held0
            qty = int(min(qty, prev_pos))
            if qty < 1:
                return False
    
            # --- compute profit vs *pre-sale* avg cost ---
            avg_cost_before = _avg_cost()
            notional        = qty * next_price
            fee_cash        = notional * fee_rate
    
            balance      += (notional - fee_cash)
            shares_held0 -= qty
    
            # proportional cost-basis adjustment
            cost_removed      = avg_cost_before * qty
            self._bought_qty  = max(0, int(getattr(self, "_bought_qty", 0)) - qty)
            self._bought_val  = max(0.0, float(getattr(self, "_bought_val", 0.0)) - cost_removed)
            self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else None
    
            # trackers
            self.total_shares_sold1.append(qty)
            self.total_sales_value1.append(notional)
            self._sold_qty = getattr(self, "_sold_qty", 0) + qty
            self._sold_val = getattr(self, "_sold_val", 0.0) + notional
    
            self.Last_sell_price = max(next_price, 0.0)
    
            # (b) NEW: compute profit vs original entry if available, else avg cost
            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            if entry_price > 0.0:
                self._close_pnl_for_bonus = (next_price - entry_price) / max(entry_price, 1e-9)
            else:
                self._close_pnl_for_bonus = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
    
            # (B) NEW: realized pnl for THIS SELL STEP (weighted by fraction sold)
            # Use avg_cost_before (pre-sale) so it corresponds to what you actually realized.
            pnl_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
            frac_sold = qty / max(prev_pos, 1e-9)
            self._realized_pnl_pct_step += float(pnl_pct) * float(frac_sold)
    
            # detect "fully closed at profit" (using avg_cost_before to avoid weirdness)
            if prev_pos > 0:
                profit_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
                if (qty == prev_pos) and (profit_pct > 0.0):
                    full_close_profit = True
    
            return True
    
        can_buy_one = balance >= (next_price * (1.0 + fee_rate))
        have_pos    = (shares_held0 > 0)
    
        # -------------------------------------------------------
        # HOLD
        # -------------------------------------------------------
        if action == 0:
            pass
    
        # -------------------------------------------------------
        # BUY (1,2,3) – direct volume
        # -------------------------------------------------------
        elif action in (1, 2, 3):
            if not can_buy_one:
                invalid_flag = True
                self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
            else:
                # (a) NEW: track position transition 0 -> >0
                prev_pos0 = shares_held0
    
                frac = float(BUY_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                cash_to_use = balance * frac
    
                denom = next_price * (1.0 + fee_rate)
                n_max = int(cash_to_use / max(denom, 1e-12))
    
                if n_max >= 1:
                    notional = n_max * next_price
                    fee_cash = notional * fee_rate
    
                    balance      -= (notional + fee_cash)
                    shares_held0 += n_max
    
                    # trackers
                    self.total_shares_bought1.append(n_max)
                    self.total_bought_value1.append(notional)
                    self._bought_qty = getattr(self, "_bought_qty", 0) + n_max
                    self._bought_val = getattr(self, "_bought_val", 0.0) + notional
    
                    self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else next_price
    
                    executed_buy = True
                    self._invalid_streak = 0
    
                    # (a) NEW: mark fresh entry on 0 -> >0
                    if prev_pos0 == 0 and shares_held0 > 0:
                        self.position_open_step    = int(getattr(self, "steps_since_reset", 0))
                        self.entry_price_for_bonus = float(next_price)
                        self._hold_steps_since_buy = 0
    
                else:
                    invalid_flag = True
                    self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
    
        # -------------------------------------------------------
        # SELL (4,5,6) – direct volume
        # -------------------------------------------------------
        elif action in (4, 5, 6):
            # UPDATED: only gate on actual position, NOT _bought_qty
            if shares_held0 <= 0:
                invalid_flag = True
            else:
                frac = float(SELL_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                qty  = max(1, int(shares_held0 * frac))
                sold_ok = _apply_sale(qty)
                if sold_ok:
                    executed_sell = True
                    self._invalid_streak = 0
                else:
                    invalid_flag = True
    
        # -------------------------------------------------------
        # Write back state & reward
        # -------------------------------------------------------
        self.balance = balance
        self.shares_held[0] = shares_held0
        self.net_worth = balance + shares_held0 * next_price
    
        # -------------------------------------------------------
        # CLEAN REWARD SHAPING (with close-position bonus)
        # -------------------------------------------------------
    
        # 1) Base reward: fractional net-worth change (clipped)
        profit_raw = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)
        profit_reward = max(min(profit_raw, 0.12), -0.12)
        r = profit_reward
    
        ###################################################################################################
        # 1b) ATR-normalized (risk-adjusted) component
        ATR_FLOOR = float(getattr(self, "ATR_FLOOR", 0.003))
        ATR_CEIL  = float(getattr(self, "ATR_CEIL", 0.05))
    
        atr_eff = min(max(atr_pct_safe, ATR_FLOOR), ATR_CEIL)
        risk_adj = profit_raw / atr_eff
        risk_adj = max(min(risk_adj, 4.0), -4.0)
    
        RISK_ADJ_WEIGHT = float(getattr(self, "RISK_ADJ_WEIGHT", 0.0015))
        r += RISK_ADJ_WEIGHT * risk_adj
        ###################################################################################################
    
        # (B) NEW: realized PnL reward on EVERY sell (partial included)
        # This is the missing “since buy → sell now” learning signal.
        REALIZED_SELL_W = float(getattr(self, "REALIZED_SELL_W", 0.004))
#z801        REALIZED_SELL_W = float(getattr(self, "REALIZED_SELL_W", 0.002))
        if executed_sell and not invalid_flag:
            rp = float(getattr(self, "_realized_pnl_pct_step", 0.0))  # weighted pct
            rp_adj = rp / max(atr_eff, 1e-6)                          # normalize by risk regime
            rp_adj = max(min(rp_adj, 3.0), -3.0)                      # clamp
            # bounded contribution
            r += REALIZED_SELL_W * float(np.tanh(rp_adj))
            
            # --- INSERT SELL DISCOVERY BONUS HERE ---
#802           self._sells_this_episode = getattr(self, "_sells_this_episode", 0)
#802            self._sells_this_episode += 1
            
            # Reward the first 15 successful sells to encourage "discovery"
#802            if self._sells_this_episode <= 15:
#802                # Use a decaying bonus: 0.005, 0.0025, 0.0016, etc.
#802                discovery_hit = 0.005 / self._sells_this_episode
#802                r += discovery_hit
            # -----------------------------------------    
        
        # 2) Per-trade turnover cost (valid trades only)
        ACTION_LOAD = {
            0: 0.00,
            1: 1.10, 2: 1.25, 3: 1.40,
            4: 0.30, 5: 0.45, 6: 0.60
        }
        if (executed_buy or executed_sell) and not invalid_flag:
            load = ACTION_LOAD.get(action, 0.0)
            r -= trade_penalty * load
    
        # 3) Flip penalty – tame B/S/B churn
        FLIP_PENALTY = float(getattr(self, "FLIP_PENALTY", 0.001))
        closing_trade = (
            (prev_action in (1, 2, 3) and action in (4, 5, 6)) or
            (prev_action in (4, 5, 6) and action in (1, 2, 3))
        )
        if (action != prev_action) and (executed_buy or executed_sell) and not invalid_flag:
            if not closing_trade:
                r -= FLIP_PENALTY

        # 4) Invalid action penalty (restore symmetry; stop sell-spam)
        if invalid_flag:
            self._invalid_streak = min(self._invalid_streak + 1, 4)
            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
            r -= min(INVALID_POST_PEN * bump, INVALID_PEN_MAX)
            self._inv_ct = getattr(self, "_inv_ct", 0) + 1
        else:
            self._invalid_streak = 0
            
        # 4) Invalid action penalty (unchanged)
#z801        if invalid_flag:
#z801            self._invalid_streak = min(self._invalid_streak + 1, 4)
#z801            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
#z801            r -= min(INVALID_POST_PEN * bump, INVALID_PEN_MAX)
#z801            self._inv_ct = getattr(self, "_inv_ct", 0) + 1
    
        # 5) Gentle risk term – penalize being very long in high ATR
        RISK_LAMBDA = float(getattr(self, "RISK_LAMBDA", 0.0005))
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac_step = float((shares_held0 * next_price) / denom_nw)
        r -= RISK_LAMBDA * pos_frac_step * atr_pct_safe
    
        # 5b) NEW: small bonus for holding winners (nudges HOLD vs early SELL)
        if (action == 0) and (shares_held0 > 0) and (profit_raw > 0) and not invalid_flag:
            r += 0.0002 * min(2.0, profit_raw * 6.0)
    
        # 6) Patience bonus in very low volatility for holding
        LOW_VOL_HOLD_ATR = float(getattr(self, "LOW_VOL_HOLD_ATR", 0.0025))
        HOLD_BONUS       = float(getattr(self, "HOLD_BONUS", 0.0001))
        if (action == 0) and not (executed_buy or executed_sell) and (atr_pct_safe < LOW_VOL_HOLD_ATR) and not invalid_flag:
            r += HOLD_BONUS
    
        # 7) Profit-sensitive de-risking bonus (partial trims in profit)
        if executed_sell and not invalid_flag and atr_pct_safe > 0.004:
            pos_frac_now = pos_frac_step
            if pos_frac_now > 0.40:
                SELL_BONUS = float(getattr(self, "SELL_BONUS", 0.0003))
                r += SELL_BONUS
    
        # (c) NEW: Bonus for fully closing a profitable position (ties SELL to BUY entry)
        CLOSE_PROFIT_BONUS = float(getattr(self, "CLOSE_PROFIT_BONUS", 0.004))
        if full_close_profit and not invalid_flag:
            pnl = float(getattr(self, "_close_pnl_for_bonus", 0.0))
            if pnl > 0.0:
                r += CLOSE_PROFIT_BONUS * min(2.0, pnl * 10.0)
    
            # reset entry markers on full close
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # (A) FIX: reset entry markers on ANY full flatten (profit OR loss)
        # Without this, "perf_since_entry" and "bars_since_entry" can become stale after closing at a loss
        # or after multi-step partial sells that eventually flatten.
        if int(shares_held0) == 0:
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # store reward + prev action
        self.rewardt = r
        self._prev_action = action
    
        # track best NW
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
        # ---------- consistency snap for trackers ----------
        if getattr(self, "_bought_qty", 0) != self.shares_held[0]:
            self._bought_qty = int(self.shares_held[0])
            if self._bought_qty == 0:
                self._bought_val = 0.0
                self.Last_buy_price = None
            else:
                if self.Last_buy_price is not None and self.Last_buy_price > 0:
                    self._bought_val = self.Last_buy_price * self._bought_qty
    
        # ---------- histories ----------
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac = float((shares_held0 * next_price) / denom_nw)
        if hasattr(self, "pos_hist"):
            self.pos_hist.append(pos_frac)
        if hasattr(self, "act_hist"):
            a = 0
            if executed_sell: a = 1
            elif executed_buy: a = 2
            self.act_hist.append(a)
    
        # ------------------------------------------------------------------
        # FIXED trade-lifecycle timers (makes "since buy" learnable)
        # ------------------------------------------------------------------
        shares_now = int(shares_held0)
    
        # steps since last trade (buy OR sell)
        self._steps_since_trade = int(getattr(self, "_steps_since_trade", 0))
        if executed_buy or executed_sell:
            self._steps_since_trade = 0
        else:
            self._steps_since_trade += 1
    
        # last trade type: +1 buy, -1 sell, 0 none
        if executed_buy:
            self._last_trade_type = 1
        elif executed_sell:
            self._last_trade_type = -1
        else:
            self._last_trade_type = int(getattr(self, "_last_trade_type", 0))
    
        # steps since ENTRY: only counts while in position
        # (you already set _hold_steps_since_buy = 0 on fresh entry)
        if shares_now > 0:
            if not executed_buy:  # don't increment on the entry bar
                self._hold_steps_since_buy = int(getattr(self, "_hold_steps_since_buy", 0)) + 1
        else:
            self._hold_steps_since_buy = 0
    
        self.steps_since_reset += 1
    
        # ---------- trade activity EMA (for state channel 15 when enabled) ----------
        trade_flag = 1.0 if (executed_buy or executed_sell) else 0.0
        alpha = getattr(self, "TRADE_EMA_ALPHA", 0.1)
        prev_ema = getattr(self, "_trade_activity_ema", 0.0)
        self._trade_activity_ema = (1.0 - alpha) * prev_ema + alpha * trade_flag


    def _take_action_011626(self, action):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
        """
    
        # ---------- fast locals ----------
        balance        = self.balance
        next_price     = self.nxtpri
        current_price  = self.curpri
        shares_held0   = self.shares_held[0]
    
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
        # UPDATED: trade penalty scale default down 10x (was 0.006)
        base_penalty   = float(getattr(self, "TRADE_PENALTY", 0.0006))
        zscale         = float(getattr(self, "ZSCALE", 8.0))  # kept for consistency if you use it elsewhere
    
        # simple volume fractions
        BUY_FRACS  = {1: 0.10, 2: 0.25, 3: 0.50}
        SELL_FRACS = {4: 0.10, 5: 0.25, 6: 0.50}
    
        # ---------- ATR-aware trade penalty ----------
        atr_pct_safe = max(self._atr_pct_cur, 1e-6)
        anneal = 0.5 + 0.5 * min(1.0, self.steps_since_reset / 64.0)
        atr_ref = 0.1
        beta = 0.25
        # UPDATED: penalty increases with ATR (was inverted)
        atr_weight = (atr_pct_safe / atr_ref) ** beta
        atr_weight = float(max(0.9, min(atr_weight, 1.2)))
        trade_penalty = base_penalty * anneal * atr_weight
    
        # ---------- invalid action penalty ----------
        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.05))
        INVALID_PEN_MAX  = float(getattr(self, "INVALID_PEN_MAX", 0.20))
        self._invalid_streak = getattr(self, "_invalid_streak", 0)
        invalid_flag = False
    
        # previous action (for flip penalty)
        prev_action = getattr(self, "_prev_action", 0)
    
        # price trace & net-worth before
        self.cprice.append(next_price)
        net_worth_before = balance + shares_held0 * current_price
    
        executed_buy  = False
        executed_sell = False
        full_close_profit = False   # did we fully close a profitable position?
    
        # NEW (b/c): per-trade close PnL buffer (used for close bonus)
        self._close_pnl_for_bonus = 0.0
    
        #########################################################################################
        self.ATR_FLOOR = 0.003        # 0.3% effective minimum ATR
        self.ATR_CEIL  = 0.05         # 5% effective maximum ATR (optional)
        self.RISK_ADJ_WEIGHT = 0.0015
    
        # ---------- helpers ----------
        def _avg_cost():
            if getattr(self, "_bought_qty", 0) > 0:
                return self._bought_val / max(self._bought_qty, 1e-12)
            lbp = getattr(self, "Last_buy_price", None)
            return lbp if (lbp is not None and lbp > 0.0) else next_price
    
        def _apply_sale(qty):
            nonlocal balance, shares_held0, full_close_profit
    
            if qty <= 0:
                return False
    
            prev_pos = shares_held0
            qty = int(min(qty, prev_pos))
            if qty < 1:
                return False
    
            # --- compute profit vs *pre-sale* avg cost ---
            avg_cost_before = _avg_cost()
            notional        = qty * next_price
            fee_cash        = notional * fee_rate
    
            balance      += (notional - fee_cash)
            shares_held0 -= qty
    
            # proportional cost-basis adjustment
            cost_removed      = avg_cost_before * qty
            self._bought_qty  = max(0, int(getattr(self, "_bought_qty", 0)) - qty)
            self._bought_val  = max(0.0, float(getattr(self, "_bought_val", 0.0)) - cost_removed)
            self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else None
    
            # trackers
            self.total_shares_sold1.append(qty)
            self.total_sales_value1.append(notional)
            self._sold_qty = getattr(self, "_sold_qty", 0) + qty
            self._sold_val = getattr(self, "_sold_val", 0.0) + notional
    
            self.Last_sell_price = max(next_price, 0.0)
    
            # (b) NEW: compute profit vs original entry if available, else avg cost
            entry_price = float(getattr(self, "entry_price_for_bonus", 0.0) or 0.0)
            if entry_price > 0.0:
                self._close_pnl_for_bonus = (next_price - entry_price) / max(entry_price, 1e-9)
            else:
                self._close_pnl_for_bonus = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
    
            # detect "fully closed at profit" (using avg_cost_before to avoid weirdness)
            if prev_pos > 0:
                profit_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
                if (qty == prev_pos) and (profit_pct > 0.0):
                    full_close_profit = True
    
            return True
    
        can_buy_one = balance >= (next_price * (1.0 + fee_rate))
        have_pos    = (shares_held0 > 0)
    
        # -------------------------------------------------------
        # HOLD
        # -------------------------------------------------------
        if action == 0:
            pass
    
        # -------------------------------------------------------
        # BUY (1,2,3) – direct volume
        # -------------------------------------------------------
        elif action in (1, 2, 3):
            if not can_buy_one:
                invalid_flag = True
                self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
            else:
                # (a) NEW: track position transition 0 -> >0
                prev_pos0 = shares_held0
    
                frac = float(BUY_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                cash_to_use = balance * frac
    
                denom = next_price * (1.0 + fee_rate)
                n_max = int(cash_to_use / max(denom, 1e-12))
    
                if n_max >= 1:
                    notional = n_max * next_price
                    fee_cash = notional * fee_rate
    
                    balance      -= (notional + fee_cash)
                    shares_held0 += n_max
    
                    # trackers
                    self.total_shares_bought1.append(n_max)
                    self.total_bought_value1.append(notional)
                    self._bought_qty = getattr(self, "_bought_qty", 0) + n_max
                    self._bought_val = getattr(self, "_bought_val", 0.0) + notional
    
                    self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else next_price
    
                    executed_buy = True
                    self._invalid_streak = 0
    
                    # (a) NEW: mark fresh entry on 0 -> >0
                    if prev_pos0 == 0 and shares_held0 > 0:
                        self.position_open_step    = int(getattr(self, "steps_since_reset", 0))
                        self.entry_price_for_bonus = float(next_price)
                        self._hold_steps_since_buy = 0
    
                else:
                    invalid_flag = True
                    self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
    
        # -------------------------------------------------------
        # SELL (4,5,6) – direct volume
        # -------------------------------------------------------
        elif action in (4, 5, 6):
            # UPDATED: only gate on actual position, NOT _bought_qty
            if shares_held0 <= 0:
                invalid_flag = True
            else:
                frac = float(SELL_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                qty  = max(1, int(shares_held0 * frac))
                sold_ok = _apply_sale(qty)
                if sold_ok:
                    executed_sell = True
                    self._invalid_streak = 0
                else:
                    invalid_flag = True
    
        # -------------------------------------------------------
        # Write back state & reward
        # -------------------------------------------------------
        self.balance = balance
        self.shares_held[0] = shares_held0
        self.net_worth = balance + shares_held0 * next_price
    
        # -------------------------------------------------------
        # CLEAN REWARD SHAPING (with close-position bonus)
        # -------------------------------------------------------
    
        # 1) Base reward: fractional net-worth change (clipped)
        profit_raw = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)
        profit_reward = max(min(profit_raw, 0.12), -0.12)
        r = profit_reward
    
        ###################################################################################################
        # 1b) ATR-normalized (risk-adjusted) component
        ATR_FLOOR = float(getattr(self, "ATR_FLOOR", 0.003))
        ATR_CEIL  = float(getattr(self, "ATR_CEIL", 0.05))
    
        atr_eff = min(max(atr_pct_safe, ATR_FLOOR), ATR_CEIL)
        risk_adj = profit_raw / atr_eff
        risk_adj = max(min(risk_adj, 4.0), -4.0)
    
        RISK_ADJ_WEIGHT = float(getattr(self, "RISK_ADJ_WEIGHT", 0.0015))
        r += RISK_ADJ_WEIGHT * risk_adj
        ###################################################################################################
    
        # 2) Per-trade turnover cost (valid trades only)
        ACTION_LOAD = {
            0: 0.00,
            1: 1.10, 2: 1.25, 3: 1.40,
            4: 0.30, 5: 0.45, 6: 0.60
        }
        if (executed_buy or executed_sell) and not invalid_flag:
            load = ACTION_LOAD.get(action, 0.0)
            r -= trade_penalty * load
    
        # 3) Flip penalty – tame B/S/B churn
        FLIP_PENALTY = float(getattr(self, "FLIP_PENALTY", 0.001))
        closing_trade = (
            (prev_action in (1, 2, 3) and action in (4, 5, 6)) or
            (prev_action in (4, 5, 6) and action in (1, 2, 3))
        )
        if (action != prev_action) and (executed_buy or executed_sell) and not invalid_flag:
            if not closing_trade:
                r -= FLIP_PENALTY
    
        # 4) Invalid action penalty (unchanged)
        if invalid_flag:
            self._invalid_streak = min(self._invalid_streak + 1, 4)
            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
            r -= min(INVALID_POST_PEN * bump, INVALID_PEN_MAX)
            self._inv_ct = getattr(self, "_inv_ct", 0) + 1
    
        # 5) Gentle risk term – penalize being very long in high ATR
        RISK_LAMBDA = float(getattr(self, "RISK_LAMBDA", 0.0005))
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac_step = float((shares_held0 * next_price) / denom_nw)
        r -= RISK_LAMBDA * pos_frac_step * atr_pct_safe
    
        # 5b) NEW: small bonus for holding winners (nudges HOLD vs early SELL)
        if (action == 0) and (shares_held0 > 0) and (profit_raw > 0) and not invalid_flag:
            r += 0.0002 * min(2.0, profit_raw * 6.0)
    
        # 6) Patience bonus in very low volatility for holding
        LOW_VOL_HOLD_ATR = float(getattr(self, "LOW_VOL_HOLD_ATR", 0.0025))
        HOLD_BONUS       = float(getattr(self, "HOLD_BONUS", 0.0001))
        if (action == 0) and not (executed_buy or executed_sell) and (atr_pct_safe < LOW_VOL_HOLD_ATR) and not invalid_flag:
            r += HOLD_BONUS
    
        # 7) Profit-sensitive de-risking bonus (partial trims in profit)
        if executed_sell and not invalid_flag and atr_pct_safe > 0.004:
            pos_frac_now = pos_frac_step
            if pos_frac_now > 0.40:
                SELL_BONUS = float(getattr(self, "SELL_BONUS", 0.0003))
                r += SELL_BONUS
    
        # (c) NEW: Bonus for fully closing a profitable position (ties SELL to BUY entry)
        CLOSE_PROFIT_BONUS = float(getattr(self, "CLOSE_PROFIT_BONUS", 0.004))
        if full_close_profit and not invalid_flag:
            pnl = float(getattr(self, "_close_pnl_for_bonus", 0.0))
            if pnl > 0.0:
                r += CLOSE_PROFIT_BONUS * min(2.0, pnl * 10.0)
    
            # reset entry markers on full close
            self.entry_price_for_bonus = 0.0
            self.position_open_step    = -1
            self._hold_steps_since_buy = 0
    
        # store reward + prev action
        self.rewardt = r
        self._prev_action = action
    
        # track best NW
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
        # ---------- consistency snap for trackers ----------
        if getattr(self, "_bought_qty", 0) != self.shares_held[0]:
            self._bought_qty = int(self.shares_held[0])
            if self._bought_qty == 0:
                self._bought_val = 0.0
                self.Last_buy_price = None
            else:
                if self.Last_buy_price is not None and self.Last_buy_price > 0:
                    self._bought_val = self.Last_buy_price * self._bought_qty
    
        # ---------- histories ----------
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac = float((shares_held0 * next_price) / denom_nw)
        if hasattr(self, "pos_hist"):
            self.pos_hist.append(pos_frac)
        if hasattr(self, "act_hist"):
            a = 0
            if executed_sell: a = 1
            elif executed_buy: a = 2
            self.act_hist.append(a)
    
        # ------------------------------------------------------------------
        # FIXED trade-lifecycle timers (makes "since buy" learnable)
        # ------------------------------------------------------------------
        shares_now = int(shares_held0)
    
        # steps since last trade (buy OR sell)
        self._steps_since_trade = int(getattr(self, "_steps_since_trade", 0))
        if executed_buy or executed_sell:
            self._steps_since_trade = 0
        else:
            self._steps_since_trade += 1
    
        # last trade type: +1 buy, -1 sell, 0 none
        if executed_buy:
            self._last_trade_type = 1
        elif executed_sell:
            self._last_trade_type = -1
        else:
            self._last_trade_type = int(getattr(self, "_last_trade_type", 0))
    
        # steps since ENTRY: only counts while in position
        # (you already set _hold_steps_since_buy = 0 on fresh entry)
        if shares_now > 0:
            if not executed_buy:  # don't increment on the entry bar
                self._hold_steps_since_buy = int(getattr(self, "_hold_steps_since_buy", 0)) + 1
        else:
            self._hold_steps_since_buy = 0
    
        self.steps_since_reset += 1
    
        # ---------- trade activity EMA (for state channel 15 when enabled) ----------
        trade_flag = 1.0 if (executed_buy or executed_sell) else 0.0
        alpha = getattr(self, "TRADE_EMA_ALPHA", 0.1)
        prev_ema = getattr(self, "_trade_activity_ema", 0.0)
        self._trade_activity_ema = (1.0 - alpha) * prev_ema + alpha * trade_flag

####################################################################################################################
    def _force_liquidate_on_done(self, price: float):
        """
        Force-close any open position at episode end to remove terminal leakage.
        Applies the same fee model as normal sells.
        """
        shares = int(self.shares_held[0])
        if shares <= 0:
            return
    
        fee_rate = float(getattr(self, "FEE_RATE", 0.005))
    
        notional = shares * float(price)
        fee_cash = notional * fee_rate
    
        # cash increases, shares go to 0
        self.balance += (notional - fee_cash)
        self.shares_held[0] = 0
    
        # keep cost-basis trackers consistent
        self._bought_qty = 0
        self._bought_val = 0.0
        self.Last_buy_price = None
    
        # reset entry anchors / timers so next episode isn't polluted
        self.entry_price_for_bonus = 0.0
        self.position_open_step    = -1
        self._hold_steps_since_buy = 0
        self._steps_since_trade    = 0
        self._last_trade_type      = 0
    
        # update net worth after liquidation
        self.net_worth = self.balance
####################################################################################################################
        
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
    
        self.current_step += 1
        self.since_last += 1
    
        # training vs eval threshold
        threshold = (
            self.date_ranges[self.rand_tick][0] - self.ndays
            if self.btraining
            else self.date_ranges[self.rand_tick][1] - self.ndays
        )
    
        # ---- TRUE TERMINAL (bankruptcy / hard fail) ----
        # tweak threshold as you like
        terminated = self.net_worth < 1000
    
        # ---- TRUNCATION (time limit / end-of-sequence) ----
        truncated = (self.since_last == 252) or (self.current_step > threshold)
    
        done = terminated or truncated
        
        # >>> THE FIX: remove terminal leakage <<<
        if done:
            # liquidate at the latest known tradable price
            # (use nxtpri since you used it for actions/valuation)
            self._force_liquidate_on_done(self.nxtpri)
    
        info = {
            "terminated": terminated,  # True only for real terminal
            "truncated": truncated,    # True for time-limit / end-of-data
        }
    
        obs = self._next_observation()
        return obs, self.rewardt, done, info

    def get_step(self):
        # Execute one time step within the environment
        return self.current_step, self.balance, self.max_net_worth, self.shares_held, self.total_shares_sold

    def set_step(self, step, balance, shares_held, shares_held_value, tick, training=False):
        # Execute one time step within the environment
        #delete training = self.btraining
        self.current_step = step
        self.balance = balance
        self.orig_bal = self.balance
        self.shares_held = shares_held
        self.rand_tick = tick

        self.profita = 0
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.total_shares_sold1 = []
        self.total_sales_value1 = []
        self.total_shares_bought1 = []
        self.total_bought_value1 = []
        # NEW: running totals (mirror the lists so render still works)
        self._bought_qty = 0
        self._bought_val = 0.0
        self._sold_qty = 0
        self._sold_val = 0.0
        self.since_last = 0
        self.rewardt = 0.
        self.cprice = []
        #  Initialize net worth history
        #delete self.bcnt = 0
        #delete if not hasattr(self, "bars_in_trade"):   self.bars_in_trade   = 0
        #delete if not hasattr(self, "peak_price"):      self.peak_price      = None
         
        tbuf = self.tbuf_train if training else self.tbuf_test 

        # --------------------------------------------
        # FAST baseline init (no _next_observation call)
        # --------------------------------------------
        data0 = tbuf.get_indices((self.rand_tick, self.current_step))[0]
        cur_price = float(data0[0])
        nxt_price = float(data0[1])
        
        # Keep env prices consistent (important for _take_action)
        self.curpri = cur_price
        self.nxtpri = nxt_price

#z802        cur_price = tbuf.get_indices((self.rand_tick, self.current_step))[0][0]
        #delete total_shares = self.balance / cur_price
        tnprice = cur_price
        self.Last_buy_price = tnprice
        self.Last_sell_price = None

        self.calibrate = 0.5
        #delete self.bs = -1.
        #delete self.xpct_investment = 1.
        self.steps_since_reset = 0 # reset warm-up counter
####################################################################################
        # NEW: reset timing markers / debug used by _take_action
        #delete self.position_open_step    = -1
        #delete self.entry_price_for_bonus = 0.0
        self._dbg_blocked_buys     = 0
        # at the end of reset() initializations
        #delete self.cooldown_until_step = -1   # no cooldown active at episode start
        # optional default; you can also put this in __init__
####################################################################################
#z901
# 9/26/25 15 TAs
        self.pos_hist = deque([0.0]* (self.ndays+1), maxlen=self.ndays+1)
        self.act_hist = deque([0]*   (self.ndays+1), maxlen=self.ndays+1)
        #delete self.bars_since_last_sell = 10**9
        #delete self.peak_price = None
####################################################################################
        self.balance = balance
        self.init_balance = float(balance)
        self._trade_activity_ema = 0.0
        self.since_last = 0
        # --- ADD: reset trade-entry tracking ---
        self.entry_price_for_bonus = 0.0
        self.position_open_step    = -1
        self._hold_steps_since_buy = 0
        self._close_pnl_for_bonus  = 0.0
        self._sells_this_episode = 0
        # --------------------------------------------
        # FIX #1 support: Buy & Hold baseline reset
        # --------------------------------------------
        # Initialize ATR proxy too (because _take_action uses self._atr_pct_cur)
        try:
            # These are z-scored inputs in your tuple
            zcls = data0[11]
            zlow = data0[17]
            zhgh = data0[16]
            self._atr_pct_cur = float(atr_proxy_percent_numba(zcls, zlow, zhgh, period=self.ATR_PERIOD))
        except Exception:
            self._atr_pct_cur = 1e-4
        
        # Baseline anchors
        p0 = cur_price if cur_price > 0.0 else (nxt_price if nxt_price > 0.0 else 1.0)
        
        self._baseline_nw0 = float(self.net_worth)
        self._baseline_entry_price = p0
        self._baseline_prev_price = p0
        self._baseline_is_init = True

    def pred_step(self, training):
        # Execute one time step within the environment
        return self._next_observation()

    def reset(self):
        # Reset the state of the environment to an initial state
        #delete training = self.btraining
        self.profita = 0
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.orig_bal = self.balance
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = [0, 0]
        self.total_shares_sold1 = []
        self.total_sales_value1 = []
        self.total_shares_bought1 = []
        self.total_bought_value1 = []
        # NEW: running totals (mirror the lists so render still works)
        self._bought_qty = 0
        self._bought_val = 0.0
        self._sold_qty = 0
        self._sold_val = 0.0
        self.since_last = 0
        self.rewardt = 0.
        self.cprice = []
        #  Initialize net worth history
        #delete self.bcnt = 0
        #delete if not hasattr(self, "bars_in_trade"):   self.bars_in_trade   = 0
        #delete if not hasattr(self, "peak_price"):      self.peak_price      = None
        
        # Set the current step to a random point within the data frame
        self.rand_tick = self.ticks[np.random.randint(len(self.ticks))]

        tbuf = self.tbuf_train if self.btraining else self.tbuf_test
        train_range, test_range = self.date_ranges[self.rand_tick][0] - self.ndays, self.date_ranges[self.rand_tick][1] - self.ndays
        self.current_step = random.randint(self.ndays+1, train_range) if self.btraining else random.randint(self.ndays+1, test_range)

        # --------------------------------------------
        # FAST baseline init (no _next_observation call)
        # --------------------------------------------
        data0 = tbuf.get_indices((self.rand_tick, self.current_step))[0]
        cur_price = float(data0[0])
        nxt_price = float(data0[1])
        
        # Keep env prices consistent (important for _take_action)
        self.curpri = cur_price
        self.nxtpri = nxt_price

#z802        cur_price = tbuf.get_indices((self.rand_tick, self.current_step))[0][0]
    
        #delete total_shares = int(self.balance / cur_price) if self.balance < self.orig_bal else 0
        tnprice = cur_price
        self.Last_buy_price = tnprice
        self.Last_sell_price = None
        self.calibrate = 0.5
        #delete self.bs = -1.
        #delete self.xpct_investment = 1.
        self.steps_since_reset = 0 # reset warm-up counter
####################################################################################
        # NEW: reset timing markers / debug used by _take_action
        #delete self.position_open_step    = -1
        #delete self.entry_price_for_bonus = 0.0
        self._dbg_blocked_buys     = 0
        # at the end of reset() initializations
        #delete self.cooldown_until_step = -1   # no cooldown active at episode start
        # optional default; you can also put this in __init__
####################################################################################
#z901
# 9/26/25 15 TAs
        self.pos_hist = deque([0.0]* (self.ndays+1), maxlen=self.ndays+1)
        self.act_hist = deque([0]*   (self.ndays+1), maxlen=self.ndays+1)
        #delete self.bars_since_last_sell = 10**9
        #delete self.peak_price = None
####################################################################################
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.init_balance = float(INITIAL_ACCOUNT_BALANCE)
        self._trade_activity_ema = 0.0
        self.since_last = 0
        # --- ADD: reset trade-entry tracking ---
        self.entry_price_for_bonus = 0.0
        self.position_open_step    = -1
        self._hold_steps_since_buy = 0
        self._close_pnl_for_bonus  = 0.0
        self._sells_this_episode = 0
        # --------------------------------------------
        # FIX #1 support: Buy & Hold baseline reset
        # --------------------------------------------
        # Initialize ATR proxy too (because _take_action uses self._atr_pct_cur)
        try:
            # These are z-scored inputs in your tuple
            zcls = data0[11]
            zlow = data0[17]
            zhgh = data0[16]
            self._atr_pct_cur = float(atr_proxy_percent_numba(zcls, zlow, zhgh, period=self.ATR_PERIOD))
        except Exception:
            self._atr_pct_cur = 1e-4
        
        # Baseline anchors
        p0 = cur_price if cur_price > 0.0 else (nxt_price if nxt_price > 0.0 else 1.0)
        
        self._baseline_nw0 = float(self.net_worth)
        self._baseline_entry_price = p0
        self._baseline_prev_price = p0
        self._baseline_is_init = True

        return self._next_observation()

    def _calculate_avg(self, total_value, total_quantity):
        return total_value / total_quantity if total_quantity > 0 else 0.0

    def _get_color(self, value):
        return '\x1b[6;30;41m' if value < 0 else '\x1b[6;30;42m'

    def _get_model_performance(self, price_trend):
        bought_value = np.sum(self.total_shares_bought1) * self.cprice[-1]
        return (self.balance + bought_value)/10000 if bought_value > 0 else self.net_worth/10000
    
    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        shares_held_total = self.shares_held[0] + self.shares_held[1]
        shares_pct0 = self.shares_held[0] / shares_held_total * 100 if shares_held_total > 0 else 0
        x = np.sum(self.total_shares_bought1)
        avg_price_bought = (np.sum(self.total_shares_bought1) / x) if x > 0 else 0.0
        x = np.sum(self.total_shares_sold1)
        avg_price_sold = (np.sum(self.total_sales_value1) /  np.sum(self.total_shares_sold1)) if x > 0 else 0.0
        profit_color = '\x1b[6;30;41m' if profit < 0 else '\x1b[6;30;42m'
        profita_color = '\x1b[6;30;41m' if self.profita < 0 else '\x1b[6;30;42m'
        price_trend = self.cprice[-1] / self.cprice[1] if len(self.cprice) > 1 else 0 # 2/18/24 added due to error
        model_performance = (self.balance + self.shares_held[0] * self.cprice[-1])/10000 if self.shares_held[0] > 0 else self.net_worth/10000
#z801        model_performance = (self.balance + np.sum(self.total_shares_bought1) * self.cprice[-1])/10000 if np.sum(self.total_shares_bought1) > 0 else self.net_worth/10000
        mperf_color = '\x1b[6;30;42m' if (price_trend / model_performance) < 1 else '\x1b[6;30;41m'
        print(f'Tick: {self.rand_tick}  step: {self.current_step} steps: {self.since_last} Hold: {shares_held_total:,.0f} Sold: {(np.sum(self.total_shares_sold1)):,.0f} Bought: {np.sum(self.total_shares_bought1):,.0f}\nBalance: {self.balance:,.2f} Net worth: {self.net_worth:,.2f} Max worth: {self.max_net_worth:,.2f} total_sales: {np.sum(self.total_sales_value1):,.2f}\nAvg price bought: {avg_price_bought:.2f} Avg price_sold: {avg_price_sold:.2f}\n{price_trend:,.2f} vs {mperf_color}{model_performance:,.2f}\x1b[0m model performance\n')


class StockTradingEnvBB_old(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataenv, sd, ndays=16):
        super(StockTradingEnvBB, self).__init__()
        
        np.random.seed(2021 + sd)
        random.seed(2021 + sd)
        self.ndays = ndays
        
        # Preallocations for Numba to fill:
        self._calib_features = np.empty((9, ndays),  dtype=np.float32)
        self._calib_aux      = np.empty((4, ndays),  dtype=np.float32)
        self._ts_flat        = np.empty(10*(ndays+1), dtype=np.float32)
        self._obs            = np.empty((16, ndays+1, 1), dtype=np.float32)
#z999        self._obs            = np.empty((10, ndays+1, 1), dtype=np.float32)

        self.tbuf_train, self.tbuf_test = dataenv.tbuf_train, dataenv.tbuf_test
        self.ticks = dataenv.ticks
        self.date_ranges = dataenv.date_ranges
        # Define observation space (example)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,ndays+1,1), dtype=np.float32)
#        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,25,1), dtype=np.float32)
#999        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,25,1), dtype=np.float32)
#z802        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # Define action space (example)
        self.action_space = spaces.Discrete(7)
#999        self.action_space = spaces.Discrete(5)
        self.btraining = True
        self.rate  = np.array([1/3, 1/2, 1.], dtype=np.float32)
        self.ratey = np.array([1/3, 1/3, 1/3], dtype=np.float32)
#################################################################################################################
        # NEW: fees/penalties & reward knobs used by _take_action
        self.FEE           = 0.0          # fixed fee per trade (set >0 if desired)
        self.FEE_RATE      = 0.0          # proportional fee (e.g., 0.0005 = 5 bps)
        self.TRADE_PENALTY = 0.012 #0.01
        self.ZSCALE        = 8.0
        
        # NEW: profit-convexity (hold winners) & round-trip friction (short holds)
        self.PROFIT_CONVEXITY_K = 0.07# 0.06
        self.PC_MAX_BONUS       = 0.003
        self.RT_MIN_HOLD        = 12
        self.RT_PENALTY_MULT    = 3.0
        
        # NEW: timing markers used by _take_action
        self.position_open_step    = -1
        self.entry_price_for_bonus = 0.0
        
        # NEW: optional debug counter
        self._dbg_blocked_buys = 0
        # optional default; you can also put this in __init__
        self.COOLDOWN_STEPS = getattr(self, "COOLDOWN_STEPS", 2)

#################################################################################################################
        self.ATR_PERIOD   = int(getattr(self, "ATR_PERIOD", 14))
        self._atr_pct_cur = 1e-4  # cached per-step ATR in percent space
#################################################################################################################
        # toggle for new vs old state block
        self.USE_NEW_STATE_BLOCK = True   # set to True to use new 6-channel state
        
        # initial balance used for nw_log_ratio; set this when you reset
        self.init_balance = 10000.0        # or override in reset()
        
        # episode length hint (used for time_in_ep_frac)
        self.MAX_EP_STEPS = 252
        
        # EMA for trade activity
        self._trade_activity_ema = 0.0
        self.TRADE_EMA_ALPHA = 0.1
        self.CLOSE_PROFIT_BONUS = 0.004
    
    def _next_observation(self):
        tbuf = self.tbuf_train if self.btraining else self.tbuf_test
        data = tbuf.get_indices((self.rand_tick, self.current_step))[0]
        
        (curcls, nxtcls, curbma, curbul, curbll, 
         zbma, zbul, zbll, zkma, zkul, zkll, 
         zcls, zrsi, zadi, zpdi, zndi, 
         zhgh, zlow, zopn, zem12, zem26) = data
    
        # ---- prices for env state ----
        self.curpri = curcls
        self.nxtpri = nxtcls
    
        # ---- cache ATR proxy (percent domain) for this step ----
        try:
            self._atr_pct_cur = atr_proxy_percent_numba(
                zcls, zlow, zhgh, period=self.ATR_PERIOD
            )
        except Exception:
            self._atr_pct_cur = 1e-4  # conservative fallback
    
        # ---- your existing "buysell" scalar (keep as-is) ----
        buysell = 2 * self.balance / (self.balance + self._bought_qty * curcls)
        # buysell = self.balance / (self.balance + self._bought_qty * curcls)
    
        # ---- base obs from your numba fn (10 TA channels expected) ----
        obs_base = compute_observation(
            zcls, zopn, zlow, zhgh, zbma, zbul, zbll, zem12, zem26,
            zpdi, zndi, zadi, zrsi,
            float(self.calibrate), 1.0,
            float(buysell), self.ndays
        )  # shape: [10, T1, 1]
        obs_base = obs_base.astype(np.float32)
        T1 = int(obs_base.shape[1])  # T+1
    
        # ============================================================
        # TOGGLE: old vs new state block
        #   - USE_NEW_STATE_BLOCK = False  -> old behavior
        #   - USE_NEW_STATE_BLOCK = True   -> new 6-channel Markov-ish state
        # ============================================================
        if getattr(self, "USE_NEW_STATE_BLOCK", False):
            # --------------------------------------------------------
            # NEW 6-CHANNEL STATE BLOCK
            # --------------------------------------------------------
            price = float(curcls)
            shares = float(self.shares_held[0])
            position_value = shares * price
    
            net_worth = self.balance + position_value
            denom_nw = max(net_worth, 1e-9)
    
            # 1) position & cash fractions [0,1]
            pos_frac  = position_value / denom_nw
            cash_frac = self.balance / denom_nw
    
            # 2) unrealized PnL (squashed)
            if getattr(self, "_bought_qty", 0) > 0 and getattr(self, "Last_buy_price", None):
                cost_basis = max(float(self.Last_buy_price), 1e-9)
                unreal_pnl_pct = (price - cost_basis) / cost_basis
            else:
                unreal_pnl_pct = 0.0
            unreal_pnl_norm = float(np.tanh(unreal_pnl_pct * 2.0))
    
            # 3) net-worth log ratio vs initial balance
            init_bal = float(getattr(self, "init_balance", 10000.0))
            nw_ratio = net_worth / max(init_bal, 1e-9)
            nw_log_ratio = float(np.clip(np.log(nw_ratio + 1e-8), -2.0, 2.0))
    
            # 4) episode progress [0,1]
            steps_in_ep = float(getattr(self, "since_last", 0))
            max_ep_steps = float(getattr(self, "MAX_EP_STEPS", 252))
            time_in_ep_frac = steps_in_ep / max(max_ep_steps, 1.0)
            time_in_ep_frac = float(np.clip(time_in_ep_frac, 0.0, 1.0))
    
            # 5) smoothed trading activity [0,1]
            trade_ema = float(getattr(self, "_trade_activity_ema", 0.0))
            trade_activity_ema = float(np.clip(trade_ema, 0.0, 1.0))
    
            # pack to (6, T1, 1)
            state_vec = np.array([
                pos_frac,
                cash_frac,
                unreal_pnl_norm,
                nw_log_ratio,
                time_in_ep_frac,
                trade_activity_ema
            ], dtype=np.float32)  # (6,)
    
            state_block = np.tile(
                state_vec[:, None, None],   # (6, 1, 1)
                (1, T1, 1)                  # -> (6, T1, 1)
            ).astype(np.float32)
    
            obs = np.concatenate([obs_base, state_block], axis=0)  # [16, T1, 1]
    
        else:
            # --------------------------------------------------------
            # OLD 5-STATE + RAMP (your current behavior)
            # --------------------------------------------------------
            # lazy init state buffers (length T1 so time lines up)
            if not hasattr(self, "pos_hist") or getattr(self, "_state_buf_len", None) != T1:
                self.pos_hist  = deque([0.0]*T1, maxlen=T1)   # position fraction history [0..1]
                self.act_hist  = deque([0]*T1,   maxlen=T1)   # 0=hold, 1=sell, 2=buy
                self._state_buf_len = T1
    
            pos_arr = np.asarray(self.pos_hist, dtype=np.float32)             # [T1]
            in_pos  = (pos_arr > 0.0).astype(np.float32)                      # [T1]
    
            # one-hot last actions per bar
            act_idx = np.asarray(self.act_hist, dtype=np.int64)               # [T1]
            act_idx = np.clip(act_idx, 0, 2)                                  # map 3->2 (buy max as buy)
            act_oh  = np.eye(3, dtype=np.float32)[act_idx]                    # [T1, 3]
            act_hold = act_oh[:, 0]
            act_sell = act_oh[:, 1]
            act_buy  = act_oh[:, 2]
    
            # stack state channels to [5, T1, 1]
            state_stack = np.stack(
                [in_pos, pos_arr, act_hold, act_sell, act_buy], axis=0
            ).astype(np.float32)[:, :, None]
    
            # positional ramp (oldest->newest = 0..1)
            ramp = np.linspace(0.0, 1.0, T1, dtype=np.float32)[None, :, None]  # [1, T1, 1]
    
            # final obs: 10 TA + 5 state + 1 ramp = 16
            obs = np.concatenate([obs_base, state_stack, ramp], axis=0)  # [16, T1, 1]
    
        # cache & return
        self._obs[...] = obs
        return obs
    
    """
    a) At sell phase and when trend is low-low, take buy action 
        with reward = -momentum * zscale. 
    b) At buy phase and when trend is high-high followed by at least nstep low-low, take sell action
        with reward = (net_worth - net_worth_before) / net_worth_before * zscale 

    T: ++---+-------++---+--------++++++---+----++++
    
    A: hhhhhhhhhhhhhhhhhhhhhhhhhhhBhhhhShhhhhhhhBhhh
    A: hhhhhhhhhhhhhhhhhhhhhhhhhhhbhhhhShhhhhhhhBhhh
    R: --+++-+++++++--+++-+++++++++++++++++-++++++++
    
    A: hhhhhhBhhhhhhhhhhhhhhhhhhhhBhhhhShhhhhhhhBhhh
    R: --+++-+++++++--+++-+++++++++++++++++-++++++++
    
                    T = Trend, R = Reward, A = Action
    """
######################################################################################################################################################
    def _take_action(self, action):
        """
        Discrete action space:
          0 = hold
          1 = buy small   (~10% cash)
          2 = buy medium  (~25% cash)
          3 = buy large   (~50% cash)
          4 = sell small  (~10% position)
          5 = sell medium (~25% position)
          6 = sell large  (~50% position)
        """
    
        # ---------- fast locals ----------
        balance        = self.balance
        price          = self.curpri
        shares_held    = self.shares_held[0]
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
    
        reward = 0.0
        traded = False
    
        # ---------- BUY ----------
        if action in (1, 2, 3):
            frac = {1: 0.10, 2: 0.25, 3: 0.50}[action]
            invest = balance * frac
    
            if invest > 0.0:
                shares = invest / price
                cost   = invest * (1.0 + fee_rate)
    
                if cost <= balance:
                    self.balance          -= cost
                    self.shares_held[0]   += shares
                    self.avg_price[0]      = price if shares_held == 0 else self.avg_price[0]
                    traded = True
    
                    # Small asymmetric ENTRY bonus (encourages participation)
                    reward += 0.002
    
                    # Stats
                    self.stats_buys += 1
    
        # ---------- SELL ----------
        elif action in (4, 5, 6):
            frac = {4: 0.10, 5: 0.25, 6: 0.50}[action]
            sell_shares = shares_held * frac
    
            if sell_shares > 0.0:
                proceeds = sell_shares * price * (1.0 - fee_rate)
                self.balance        += proceeds
                self.shares_held[0] -= sell_shares
                traded = True
    
                # Realized PnL
                pnl = (price - self.avg_price[0]) * sell_shares
    
                # EXIT reward = realized profit signal
                reward += pnl / max(self.net_worth, 1e-6)
    
                # Stats
                self.stats_sells += 1
                self.total_sales += proceeds
    
        # ---------- HOLD ----------
        else:
            # Mild decay to prevent infinite holding
            reward -= 0.0005
            self.stats_holds += 1
    
        # ---------- POSITION SHAPING (dense, symmetric) ----------
        # Encourage being long when price > avg, discourage when below
        if self.shares_held[0] > 0:
            unrealized = (price - self.avg_price[0]) / self.avg_price[0]
            reward += 0.001 * unrealized
    
        # ---------- ACTION BALANCE PENALTY ----------
        # Prevent collapse into pure buy or pure hold
        if not traded:
            reward -= 0.0002
    
        return reward


    def _take_action_old(self, action: int):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
    
        Key behaviors (updated):
          - Invalid/blocked actions are true no-ops (no penalty).
          - Trade costs/penalties apply ONLY if a trade actually executes.
          - Trade penalty scales with trade size (trade_frac).
          - Optional tiny buy-entry bonus only when a BUY executes.
          - Optional hold bonus (e.g., in calm regimes) stays allowed.
          - Reward is based on change in net worth (or your existing reward logic), plus small shaping.
        """
    
        # ---------------- fast locals ----------------
        balance        = float(self.balance)
        cur_price      = float(self.curpri)   # set in _next_observation
        next_price     = float(self.nxtpri)   # set in _next_observation
        shares_held0   = float(self.shares_held[0]) if hasattr(self.shares_held, "__len__") else float(self.shares_held)
    
        # --------- tunables / defaults (safe getattr) ----------
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))        # brokerage+slippage proxy
        trade_penalty  = float(getattr(self, "TRADE_PENALTY", 0.006))   # extra regularizer
        reward_scale   = float(getattr(self, "REWARD_SCALE", 1.0))      # if you still use it downstream
        eps_cash       = float(getattr(self, "EPS_CASH", 1e-12))
        eps_price      = float(getattr(self, "EPS_PRICE", 1e-12))
        max_shares     = float(getattr(self, "MAX_SHARES", 1e18))       # optional cap
        min_trade_sh   = float(getattr(self, "MIN_TRADE_SHARES", 0.0))  # optional min lot in shares
        allow_short    = bool(getattr(self, "ALLOW_SHORT", False))      # default long-only
    
        # Fractions for (small/med/large)
        buy_fracs  = (0.10, 0.25, 0.50)
        sell_fracs = (0.10, 0.25, 0.50)
    
        # --------- (optional) "load" factor (e.g., ATR/vol gating) ----------
        # If you have your own load calc, keep it. This just provides a fallback.
        # load in [0..1] where 1 = normal trading cost/pen, 0 = very calm / encourage action
        load = float(getattr(self, "trade_cost_load", 1.0))
        if not (0.0 <= load <= 1.0):
            load = max(0.0, min(1.0, load))
    
        # --------- stats (optional) ----------
        # Create counters if not present
        if not hasattr(self, "_dbg_trades"):
            self._dbg_trades = 0
            self._dbg_buys = 0
            self._dbg_sells = 0
            self._dbg_holds = 0
            self._dbg_invalid = 0
            self._dbg_trade_dollars = 0.0
            self._dbg_fees_paid = 0.0
            self._dbg_entry_bonus = 0.0
            self._dbg_hold_bonus = 0.0
    
        # ---------------- compute current net worth ----------------
        net_worth_before = balance + shares_held0 * cur_price
    
        # ---------------- decode action to intent ----------------
        invalid = False
        executed_buy = False
        executed_sell = False
    
        dollars_traded = 0.0
        fee_paid = 0.0
        entry_bonus = 0.0
        hold_bonus = 0.0
    
        # Default: no-op
        new_balance = balance
        new_shares  = shares_held0
    
        # ---------------- HOLD ----------------
        if action == 0:
            self._dbg_holds += 1
    
            # Optional: tiny hold bonus in calm regime (keep if you already had it)
            # Example: encourage patience when volatility is low
            hold_bonus_base = float(getattr(self, "HOLD_BONUS", 0.0))  # set to e.g. 1e-4 if you want
            if hold_bonus_base != 0.0:
                # if load small (calm), bonus bigger
                hold_bonus = hold_bonus_base * (1.0 - load)
    
        # ---------------- BUY ----------------
        elif action in (1, 2, 3):
            frac = buy_fracs[action - 1]
    
            # buy power (long-only): cannot exceed cash
            buy_cash = balance * frac
            if buy_cash <= eps_cash:
                invalid = True
            else:
                # shares to buy
                price = max(cur_price, eps_price)
                sh = buy_cash / price
    
                # optional lot sizing
                if sh < min_trade_sh:
                    invalid = True
                else:
                    # cap shares if you want
                    sh = min(sh, max(0.0, max_shares - shares_held0))
                    if sh <= 0.0:
                        invalid = True
                    else:
                        # Execute buy
                        cost = sh * price
                        # Fees proportional to dollars traded
                        fee_paid = fee_rate * cost
                        total_cost = cost + fee_paid
    
                        if total_cost > balance + 1e-9:
                            # Not enough cash after fees -> clamp down
                            max_affordable = max(0.0, balance / (1.0 + fee_rate))
                            sh2 = max_affordable / price
                            if sh2 < min_trade_sh:
                                invalid = True
                            else:
                                cost2 = sh2 * price
                                fee_paid = fee_rate * cost2
                                total_cost = cost2 + fee_paid
                                sh = sh2
    
                        if not invalid:
                            new_balance = balance - total_cost
                            new_shares  = shares_held0 + sh
                            executed_buy = True
    
                            dollars_traded = sh * price
                            self._dbg_buys += 1
                            self._dbg_trades += 1
    
                            # Tiny buy-entry bonus ONLY when a BUY executes
                            # Keep this very small; optionally scale by (1-load) to reward entries in calm regimes.
                            entry_bonus_base = float(getattr(self, "BUY_ENTRY_BONUS", 0.0))  # e.g. 1e-4
                            if entry_bonus_base != 0.0:
                                entry_bonus = entry_bonus_base * (1.0 - load)
    
        # ---------------- SELL ----------------
        elif action in (4, 5, 6):
            frac = sell_fracs[action - 4]
    
            # Long-only: can only sell what you have
            if shares_held0 <= 0.0 and not allow_short:
                invalid = True
            else:
                sh = abs(shares_held0) * frac if shares_held0 != 0.0 else 0.0
    
                if sh < min_trade_sh:
                    invalid = True
                else:
                    price = max(cur_price, eps_price)
    
                    # Long-only sell
                    if not allow_short:
                        sh = min(sh, shares_held0)
                        if sh <= 0.0:
                            invalid = True
    
                    if not invalid:
                        proceeds = sh * price
                        fee_paid = fee_rate * proceeds
                        net_proceeds = proceeds - fee_paid
    
                        # Execute sell
                        new_balance = balance + net_proceeds
                        new_shares  = shares_held0 - sh
                        executed_sell = True
    
                        dollars_traded = proceeds
                        self._dbg_sells += 1
                        self._dbg_trades += 1
    
        else:
            invalid = True
    
        # ---------------- invalid action = true no-op (NO penalty) ----------------
        if invalid:
            self._dbg_invalid += 1
            new_balance = balance
            new_shares  = shares_held0
            executed_buy = executed_sell = False
            dollars_traded = 0.0
            fee_paid = 0.0
            entry_bonus = 0.0
            hold_bonus = 0.0
    
        # ---------------- apply state updates ----------------
        self.balance = new_balance
        if hasattr(self.shares_held, "__len__"):
            self.shares_held[0] = new_shares
        else:
            self.shares_held = new_shares
    
        # Keep net worth updated if you track it
        net_worth_after = float(self.balance) + float(new_shares) * next_price
        self.net_worth = net_worth_after
    
        # ---------------- base reward: change in net worth ----------------
        # This is a common stable default. If you already have your own reward, swap it in here.
        # Using next_price avoids immediate lookahead leakage as long as next_price is "next step" price.
        base_reward = (net_worth_after - net_worth_before) / max(net_worth_before, 1e-9)
    
        # ---------------- trade penalty ONLY if trade executed ----------------
        # Scale by trade_frac to avoid punishing small exploratory trades.
        trade_pen = 0.0
        if executed_buy or executed_sell:
            trade_frac = dollars_traded / max(net_worth_before, 1e-9)
            # penalty proportional to trade size and load
            trade_pen = trade_penalty * load * trade_frac
    
        # ---------------- final reward ----------------
        r = base_reward - trade_pen + entry_bonus + hold_bonus
    
        # Optional: clip raw reward (before affine normalization in loss)
        rclip = getattr(self, "REWARD_CLIP", None)
        if rclip is not None:
            r = float(max(-rclip, min(rclip, r)))
    
        # ---------------- accumulate stats ----------------
        self._dbg_trade_dollars += float(dollars_traded)
        self._dbg_fees_paid     += float(fee_paid)
        self._dbg_entry_bonus   += float(entry_bonus)
        self._dbg_hold_bonus    += float(hold_bonus)
    
        # If you track immediate reward (for render/debug)
        self.reward = float(r) * reward_scale
    
        return self.reward

    
    def _take_action_old(self, action):
        """
        Actions (pure RL, no ladder):
          0 = hold
          1 = buy  small  (~10% of cash)
          2 = buy  medium (~25% of cash)
          3 = buy  large  (~50% of cash)
          4 = sell small  (~10% of position)
          5 = sell medium (~25% of position)
          6 = sell large  (~50% of position)
        """
    
        # ---------- fast locals ----------
        balance        = self.balance
        next_price     = self.nxtpri
        current_price  = self.curpri
        shares_held0   = self.shares_held[0]
    
        fee_rate       = float(getattr(self, "FEE_RATE", 0.005))
        base_penalty   = float(getattr(self, "TRADE_PENALTY", 0.006))
        zscale         = float(getattr(self, "ZSCALE", 8.0))  # kept for consistency if you use it elsewhere
    
        # simple volume fractions
        BUY_FRACS  = {1: 0.10, 2: 0.25, 3: 0.50}
        SELL_FRACS = {4: 0.10, 5: 0.25, 6: 0.50}
    
        # ---------- ATR-aware trade penalty ----------
        atr_pct_safe = max(self._atr_pct_cur, 1e-6)
        anneal = 0.5 + 0.5 * min(1.0, self.steps_since_reset / 64.0)
        atr_ref = 0.1
        beta = 0.25
        atr_weight = (atr_ref / atr_pct_safe) ** beta
        atr_weight = float(max(0.9, min(atr_weight, 1.2)))
        trade_penalty = base_penalty * anneal * atr_weight
    
        # ---------- invalid action penalty ----------
        INVALID_POST_PEN = float(getattr(self, "INVALID_POST_PEN", 0.05))
        INVALID_PEN_MAX  = float(getattr(self, "INVALID_PEN_MAX", 0.20))
        self._invalid_streak = getattr(self, "_invalid_streak", 0)
        invalid_flag = False
    
        # previous action (for flip penalty)
        prev_action = getattr(self, "_prev_action", 0)
    
        # price trace & net-worth before
        self.cprice.append(next_price)
        net_worth_before = balance + shares_held0 * current_price
    
        executed_buy  = False
        executed_sell = False
        full_close_profit = False   # NEW: did we fully close a profitable position?
        #########################################################################################
        self.ATR_FLOOR = 0.003        # 0.3% effective minimum ATR
        self.ATR_CEIL  = 0.05         # 5% effective maximum ATR (optional)
        self.RISK_ADJ_WEIGHT = 0.0015
    
        # ---------- helpers ----------
        def _avg_cost():
            if getattr(self, "_bought_qty", 0) > 0:
                return self._bought_val / max(self._bought_qty, 1e-12)
            lbp = getattr(self, "Last_buy_price", None)
            return lbp if (lbp is not None and lbp > 0.0) else next_price
    
        def _apply_sale(qty):
            nonlocal balance, shares_held0, full_close_profit

            if qty <= 0:
                return False

            prev_pos = shares_held0
            qty = int(min(qty, prev_pos))
            if qty < 1:
                return False

            # --- compute profit vs *pre-sale* avg cost ---
            avg_cost_before = _avg_cost()
            notional        = qty * next_price
            fee_cash        = notional * fee_rate

            balance      += (notional - fee_cash)
            shares_held0 -= qty

            # proportional cost-basis adjustment
            cost_removed      = avg_cost_before * qty
            self._bought_qty  = max(0, int(getattr(self, "_bought_qty", 0)) - qty)
            self._bought_val  = max(0.0, float(getattr(self, "_bought_val", 0.0)) - cost_removed)
            self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else None

            # trackers
            self.total_shares_sold1.append(qty)
            self.total_sales_value1.append(notional)
            self._sold_qty = getattr(self, "_sold_qty", 0) + qty
            self._sold_val = getattr(self, "_sold_val", 0.0) + notional

            self.Last_sell_price = max(next_price, 0.0)

            # NEW: detect "fully closed at profit"
            if prev_pos > 0:
                profit_pct = (next_price - avg_cost_before) / max(avg_cost_before, 1e-9)
                if (qty == prev_pos) and (profit_pct > 0.0):
                    full_close_profit = True

            return True
    
        can_buy_one = balance >= (next_price * (1.0 + fee_rate))
        have_pos    = (shares_held0 > 0)
    
        # -------------------------------------------------------
        # HOLD
        # -------------------------------------------------------
        if action == 0:
            pass
    
        # -------------------------------------------------------
        # BUY (1,2,3) – direct volume
        # -------------------------------------------------------
        elif action in (1, 2, 3):
            if not can_buy_one:
                invalid_flag = True
                self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
            else:
                frac = float(BUY_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                cash_to_use = balance * frac
    
                denom = next_price * (1.0 + fee_rate)
                n_max = int(cash_to_use / max(denom, 1e-12))
    
                if n_max >= 1:
                    notional = n_max * next_price
                    fee_cash = notional * fee_rate
    
                    balance      -= (notional + fee_cash)
                    shares_held0 += n_max
    
                    # trackers
                    self.total_shares_bought1.append(n_max)
                    self.total_bought_value1.append(notional)
                    self._bought_qty = getattr(self, "_bought_qty", 0) + n_max
                    self._bought_val = getattr(self, "_bought_val", 0.0) + notional
    
                    self.Last_buy_price = (self._bought_val / self._bought_qty) if self._bought_qty > 0 else next_price
    
                    executed_buy = True
                    self._invalid_streak = 0
                else:
                    invalid_flag = True
                    self._dbg_blocked_buys = getattr(self, "_dbg_blocked_buys", 0) + 1
    
        # -------------------------------------------------------
        # SELL (4,5,6) – direct volume
        # -------------------------------------------------------
        elif action in (4, 5, 6):
            if not have_pos or getattr(self, "_bought_qty", 0) <= 0:
                invalid_flag = True
            else:
                frac = float(SELL_FRACS[action])
                frac = max(0.0, min(1.0, frac))
                qty  = max(1, int(shares_held0 * frac))
                sold_ok = _apply_sale(qty)
                if sold_ok:
                    executed_sell = True
                    self._invalid_streak = 0
                else:
                    invalid_flag = True
    
        # -------------------------------------------------------
        # Write back state & reward
        # -------------------------------------------------------
        self.balance = balance
        self.shares_held[0] = shares_held0
        self.net_worth = balance + shares_held0 * next_price
    
        # -------------------------------------------------------
        # CLEAN REWARD SHAPING (with close-position bonus)
        # -------------------------------------------------------

        # 1) Base reward: fractional net-worth change (clipped)
        profit_raw = (self.net_worth - net_worth_before) / max(net_worth_before, 1e-9)
        profit_reward = max(min(profit_raw, 0.12), -0.12)
        r = profit_reward
        ###################################################################################################
        # 1b) ATR-normalized (risk-adjusted) component
        #     Uses atr_pct_safe but protects against tiny ATR blowing up reward.
        ATR_FLOOR = float(getattr(self, "ATR_FLOOR", 0.003))   # 0.3% minimum effective ATR
        ATR_CEIL  = float(getattr(self, "ATR_CEIL", 0.05))    # optional: cap at 5% ATR

        # clamp ATR into [ATR_FLOOR, ATR_CEIL] to avoid huge rewards in ultra-low vol
        atr_eff = min(max(atr_pct_safe, ATR_FLOOR), ATR_CEIL)

        # Sharpe-like step reward: return per unit of risk
        risk_adj = profit_raw / atr_eff

        # Clip abnormal steps (gaps, weird bars, etc.)
        risk_adj = max(min(risk_adj, 4.0), -4.0)

        # Scale to keep total reward magnitudes reasonable
        RISK_ADJ_WEIGHT = float(getattr(self, "RISK_ADJ_WEIGHT", 0.0015))
        r += RISK_ADJ_WEIGHT * risk_adj
        ###################################################################################################

        # 2) Per-trade turnover cost (valid trades only)
#        ACTION_LOAD = {
#            0: 0.00,
#            1: 0.90, 2: 1.05, 3: 1.20,
#            4: 0.40, 5: 0.55, 6: 0.70
#        }
        ACTION_LOAD = {
            0: 0.00,          # hold
            1: 1.10, 2: 1.25, 3: 1.40,   # buys
            4: 0.30, 5: 0.45, 6: 0.60    # sells
        }
        if (executed_buy or executed_sell) and not invalid_flag:
            load = ACTION_LOAD.get(action, 0.0)
            r -= trade_penalty * load    # trade_penalty is ATR-aware

        # 2b) BUY entry bonus (state-based, no lookahead)
#        if executed_buy and not invalid_flag:
#            ENTRY_ATR_REF = float(getattr(self, "ENTRY_ATR_REF", 0.006))   # tune 0.004–0.010
#            ENTRY_BONUS_W = float(getattr(self, "ENTRY_BONUS_W", 0.002))   # tune 0.001–0.006
#        
#            entry_bonus = max(0.0, ENTRY_ATR_REF - atr_pct_safe)           # only when vol is low
#            r += ENTRY_BONUS_W * entry_bonus

        # 3) Flip penalty – tame B/S/B churn
        FLIP_PENALTY = float(getattr(self, "FLIP_PENALTY", 0.001))
        closing_trade = (
            (prev_action in (1, 2, 3) and action in (4, 5, 6)) or
            (prev_action in (4, 5, 6) and action in (1, 2, 3))
        )
        if (action != prev_action) and (executed_buy or executed_sell) and not invalid_flag:
            if not closing_trade:
                r -= FLIP_PENALTY

        # 4) Invalid action penalty (unchanged)
        if invalid_flag:
            self._invalid_streak = min(self._invalid_streak + 1, 4)
            bump = 1.0 + 0.25 * (self._invalid_streak - 1)
            r -= min(INVALID_POST_PEN * bump, INVALID_PEN_MAX)
            self._inv_ct = getattr(self, "_inv_ct", 0) + 1

        # 5) Gentle risk term – penalize being very long in high ATR
        RISK_LAMBDA = float(getattr(self, "RISK_LAMBDA", 0.0005))
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac_step = float((shares_held0 * next_price) / denom_nw)  # 0..1
        r -= RISK_LAMBDA * pos_frac_step * atr_pct_safe

        # 6) Patience bonus in very low volatility for holding
        LOW_VOL_HOLD_ATR = float(getattr(self, "LOW_VOL_HOLD_ATR", 0.0025))
        HOLD_BONUS       = float(getattr(self, "HOLD_BONUS", 0.0001))
        if (action == 0) and not (executed_buy or executed_sell) and (atr_pct_safe < LOW_VOL_HOLD_ATR) and not invalid_flag:
            r += HOLD_BONUS

        # 7) Profit-sensitive de-risking bonus (partial trims in profit)
        if executed_sell and not invalid_flag and atr_pct_safe > 0.004:
            pos_frac_now = pos_frac_step
            if pos_frac_now > 0.40:   # only when we were meaningfully invested
                SELL_BONUS = float(getattr(self, "SELL_BONUS", 0.0006))
                r += SELL_BONUS

        # 8) Bonus for closing entire position at profit (teaches exit logic)
        CLOSE_PROFIT_BONUS = float(getattr(self, "CLOSE_PROFIT_BONUS", 0.004))
        
        if executed_sell and not invalid_flag:
            # Did we close the entire position?
            if self.shares_held[0] == 0 and self.Last_buy_price is not None:
                # Compute realized PnL
                pnl = (next_price - self.Last_buy_price) / max(self.Last_buy_price, 1e-9)
                
                if pnl > 0:
                    # Reward proportional to profit, softly saturated
                    r += CLOSE_PROFIT_BONUS * min(2.0, pnl * 10.0)
        
        # 8) EXTRA bonus for fully closing a profitable position
#999        if full_close_profit and not invalid_flag:
#999            CLOSE_BONUS = float(getattr(self, "CLOSE_BONUS", 0.0012))
#999            r += CLOSE_BONUS

        # store reward + prev action
        self.rewardt = r
        self._prev_action = action
    
        # track best NW
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
        # ---------- consistency snap for trackers ----------
        if getattr(self, "_bought_qty", 0) != self.shares_held[0]:
            self._bought_qty = int(self.shares_held[0])
            if self._bought_qty == 0:
                self._bought_val = 0.0
                self.Last_buy_price = None
            else:
                if self.Last_buy_price is not None and self.Last_buy_price > 0:
                    self._bought_val = self.Last_buy_price * self._bought_qty
    
        # ---------- histories ----------
        denom_nw = max(self.net_worth, 1e-9)
        pos_frac = float((shares_held0 * next_price) / denom_nw)
        if hasattr(self, "pos_hist"):
            self.pos_hist.append(pos_frac)
        if hasattr(self, "act_hist"):
            a = 0
            if executed_sell: a = 1
            elif executed_buy: a = 2
            self.act_hist.append(a)
    
        # simple timers (if you still use them for logging)
        self._hold_steps_since_buy = getattr(self, "_hold_steps_since_buy", 0)
        self._steps_since_sell     = getattr(self, "_steps_since_sell", 0)
        if executed_buy:
            self._hold_steps_since_buy = 0
            self._steps_since_sell    += 1
        elif executed_sell:
            self._hold_steps_since_buy += 1
            self._steps_since_sell      = 0
        else:
            self._hold_steps_since_buy += 1
            self._steps_since_sell     += 1
    
        self.steps_since_reset += 1
    
        # ---------- trade activity EMA (for state channel 15 when enabled) ----------
        trade_flag = 1.0 if (executed_buy or executed_sell) else 0.0
        alpha = getattr(self, "TRADE_EMA_ALPHA", 0.1)
        prev_ema = getattr(self, "_trade_activity_ema", 0.0)
        self._trade_activity_ema = (1.0 - alpha) * prev_ema + alpha * trade_flag


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
    
        self.current_step += 1
        self.since_last += 1
    
        # training vs eval threshold
        threshold = (
            self.date_ranges[self.rand_tick][0] - self.ndays
            if self.btraining
            else self.date_ranges[self.rand_tick][1] - self.ndays
        )
    
        # ---- TRUE TERMINAL (bankruptcy / hard fail) ----
        # tweak threshold as you like
        terminated = self.net_worth < 1000
    
        # ---- TRUNCATION (time limit / end-of-sequence) ----
        truncated = (self.since_last == 252) or (self.current_step > threshold)
    
        done = terminated or truncated
    
        info = {
            "terminated": terminated,  # True only for real terminal
            "truncated": truncated,    # True for time-limit / end-of-data
        }
    
        obs = self._next_observation()
        return obs, self.rewardt, done, info


    def step_old(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.since_last += 1
        
        threshold = self.date_ranges[self.rand_tick][0] - self.ndays if self.btraining else self.date_ranges[self.rand_tick][1] - self.ndays
        
        terminated = self.net_worth < 1000
        truncated  = (self.since_last == 252) or (self.current_step > threshold)
        done = terminated or truncated
        info = {"terminated": terminated, "truncated": truncated}
        
        obs = self._next_observation()
        
        return obs, self.rewardt, done, info

    def get_step(self):
        # Execute one time step within the environment
        return self.current_step, self.balance, self.max_net_worth, self.shares_held, self.total_shares_sold

    def set_step(self, step, balance, shares_held, shares_held_value, tick, training=False):
        # Execute one time step within the environment
        training = self.btraining
        self.current_step = step
        self.balance = balance
        self.orig_bal = self.balance
        self.shares_held = shares_held
        self.rand_tick = tick

        self.profita = 0
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.total_shares_sold1 = []
        self.total_sales_value1 = []
        self.total_shares_bought1 = []
        self.total_bought_value1 = []
        # NEW: running totals (mirror the lists so render still works)
        self._bought_qty = 0
        self._bought_val = 0.0
        self._sold_qty = 0
        self._sold_val = 0.0
        self.since_last = 0
        self.rewardt = 0.
        self.cprice = []
        #  Initialize net worth history
        self.bcnt = 0
        if not hasattr(self, "bars_in_trade"):   self.bars_in_trade   = 0
        if not hasattr(self, "peak_price"):      self.peak_price      = None
         
        tbuf = self.tbuf_train if training else self.tbuf_test 

        cur_price = tbuf.get_indices((self.rand_tick, self.current_step))[0][0]
        total_shares = self.balance / cur_price
        tnprice = cur_price
        self.Last_buy_price = None
#z802        self.Last_buy_price = tnprice
        self.Last_sell_price = None

        self.calibrate = 0.5
        self.bs = -1.
        self.xpct_investment = 1.
        self.steps_since_reset = 0 # reset warm-up counter
####################################################################################
        # NEW: reset timing markers / debug used by _take_action
        self.position_open_step    = -1
        self.entry_price_for_bonus = 0.0
        self._dbg_blocked_buys     = 0
        # at the end of reset() initializations
        self.cooldown_until_step = -1   # no cooldown active at episode start
        # optional default; you can also put this in __init__
####################################################################################
#z901
# 9/26/25 15 TAs
        self.pos_hist = deque([0.0]* (self.ndays+1), maxlen=self.ndays+1)
        self.act_hist = deque([0]*   (self.ndays+1), maxlen=self.ndays+1)
        self.bars_since_last_sell = 10**9
        self.peak_price = None
####################################################################################
        self.balance = balance
        self.init_balance = float(balance)
        self._trade_activity_ema = 0.0
        self.since_last = 0

    def pred_step(self, training):
        # Execute one time step within the environment
        return self._next_observation()

    def reset(self):
        # Reset the state of the environment to an initial state
        training = self.btraining
        self.profita = 0
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.orig_bal = self.balance
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = [0, 0]
        self.total_shares_sold1 = []
        self.total_sales_value1 = []
        self.total_shares_bought1 = []
        self.total_bought_value1 = []
        # NEW: running totals (mirror the lists so render still works)
        self._bought_qty = 0
        self._bought_val = 0.0
        self._sold_qty = 0
        self._sold_val = 0.0
        self.since_last = 0
        self.rewardt = 0.
        self.cprice = []
        #  Initialize net worth history
        self.bcnt = 0
        if not hasattr(self, "bars_in_trade"):   self.bars_in_trade   = 0
        if not hasattr(self, "peak_price"):      self.peak_price      = None
        
        # Set the current step to a random point within the data frame
        self.rand_tick = self.ticks[np.random.randint(len(self.ticks))]

        tbuf = self.tbuf_train if training else self.tbuf_test
        train_range, test_range = self.date_ranges[self.rand_tick][0] - self.ndays, self.date_ranges[self.rand_tick][1] - self.ndays
        self.current_step = random.randint(self.ndays+1, train_range) if training else random.randint(self.ndays+1, test_range)

        cur_price = tbuf.get_indices((self.rand_tick, self.current_step))[0][0]
    
        total_shares = int(self.balance / cur_price) if self.balance < self.orig_bal else 0
        tnprice = cur_price
        self.Last_buy_price = None
#z802        self.Last_buy_price = tnprice
        self.Last_sell_price = None
        self.calibrate = 0.5
        self.bs = -1.
        self.xpct_investment = 1.
        self.steps_since_reset = 0 # reset warm-up counter
####################################################################################
        # NEW: reset timing markers / debug used by _take_action
        self.position_open_step    = -1
        self.entry_price_for_bonus = 0.0
        self._dbg_blocked_buys     = 0
        # at the end of reset() initializations
        self.cooldown_until_step = -1   # no cooldown active at episode start
        # optional default; you can also put this in __init__
####################################################################################
#z901
# 9/26/25 15 TAs
        self.pos_hist = deque([0.0]* (self.ndays+1), maxlen=self.ndays+1)
        self.act_hist = deque([0]*   (self.ndays+1), maxlen=self.ndays+1)
        self.bars_since_last_sell = 10**9
        self.peak_price = None
####################################################################################
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.init_balance = float(INITIAL_ACCOUNT_BALANCE)
        self._trade_activity_ema = 0.0
        self.since_last = 0

        return self._next_observation()

    def _calculate_avg(self, total_value, total_quantity):
        return total_value / total_quantity if total_quantity > 0 else 0.0

    def _get_color(self, value):
        return '\x1b[6;30;41m' if value < 0 else '\x1b[6;30;42m'

    def _get_model_performance(self, price_trend):
        bought_value = np.sum(self.total_shares_bought1) * self.cprice[-1]
        return (self.balance + bought_value)/10000 if bought_value > 0 else self.net_worth/10000
    
    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        shares_held_total = self.shares_held[0] + self.shares_held[1]
        shares_pct0 = self.shares_held[0] / shares_held_total * 100 if shares_held_total > 0 else 0
        x = np.sum(self.total_shares_bought1)
        avg_price_bought = (np.sum(self.total_shares_bought1) / x) if x > 0 else 0.0
        x = np.sum(self.total_shares_sold1)
        avg_price_sold = (np.sum(self.total_sales_value1) /  np.sum(self.total_shares_sold1)) if x > 0 else 0.0
        profit_color = '\x1b[6;30;41m' if profit < 0 else '\x1b[6;30;42m'
        profita_color = '\x1b[6;30;41m' if self.profita < 0 else '\x1b[6;30;42m'
        price_trend = self.cprice[-1] / self.cprice[1] if len(self.cprice) > 1 else 0 # 2/18/24 added due to error
        model_performance = (self.balance + self.shares_held[0] * self.cprice[-1])/10000 if self.shares_held[0] > 0 else self.net_worth/10000
#z801        model_performance = (self.balance + np.sum(self.total_shares_bought1) * self.cprice[-1])/10000 if np.sum(self.total_shares_bought1) > 0 else self.net_worth/10000
        mperf_color = '\x1b[6;30;42m' if (price_trend / model_performance) < 1 else '\x1b[6;30;41m'
        print(f'Tick: {self.rand_tick}  step: {self.current_step} steps: {self.since_last} Hold: {shares_held_total:,.0f} Sold: {(np.sum(self.total_shares_sold1)):,.0f} Bought: {np.sum(self.total_shares_bought1):,.0f}\nBalance: {self.balance:,.2f} Net worth: {self.net_worth:,.2f} Max worth: {self.max_net_worth:,.2f} total_sales: {np.sum(self.total_sales_value1):,.2f}\nAvg price bought: {avg_price_bought:.2f} Avg price_sold: {avg_price_sold:.2f}\n{price_trend:,.2f} vs {mperf_color}{model_performance:,.2f}\x1b[0m model performance\n')
