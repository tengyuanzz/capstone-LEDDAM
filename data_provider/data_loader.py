# File: data_provider/data_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
# from utils.timefeatures import time_features # Ensure this utility exists and is importable

class Dataset_Simplified_Finance(Dataset):

    def __init__(self, root_path, technical_indicators, rolling_window, flag='train', size=None,
        features='S', data_path='AAPL_xlstm.csv', # Specify your data file
        target='Close',   # Specify your target column name
        scale=True):

        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            # Set default lengths if not provided (adjust as needed)
            self.seq_len = 96
            self.label_len = 48 # This might not be used if only seq_y is needed for loss
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1] # Store it, though __getitem__ below doesn't use it explicitly for slicing seq_y differently
            self.pred_len = size[2]
        # init type
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Store configuration
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.technical_indicators = technical_indicators
        self.rolling_window = rolling_window

        # Load and process data
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)

        # --- Load Data ---
        # Assuming the first column is date and the rest are numerical features (like OHLCV)
        # Adjust read_csv parameters if needed (e.g., separator, header)
        df_raw = pd.read_csv(file_path)
        if self.data_path == 'AAPL_xlstm_sentiment_comb.csv':
            df_raw.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "w_neutral","w_positive","w_negative"]
        else:
            df_raw.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if self.technical_indicators:
            # df_raw = df_raw.rolling(self.rolling_window).mean()
            df_raw['MA'] = df_raw["Close"].rolling(window=self.rolling_window).mean()
            df_raw['RSI'] = RSIIndicator(df_raw["Close"]).rsi()
            macd = MACD(df_raw["Close"])
            df_raw['MACD'] = macd.macd()
            bollinger = BollingerBands(df_raw["Close"])
            df_raw['Bollinger_high'] = bollinger.bollinger_hband()
            df_raw['Bollinger_low'] = bollinger.bollinger_lband()
            df_raw = df_raw.dropna(inplace=False).reset_index(drop=True)
            df_raw = df_raw.dropna(how='any', axis=0).drop(columns=['Date'])

        self.logged_cols = []
        for i, col in enumerate(df_raw.columns):
            if col not in ["Date", "Volume", "MACD", "w_negative", "w_positive", "w_neutral"]:
                self.logged_cols.append(i)
                df_raw[col] = np.log(df_raw[col])

        # --- Define Chronological Split Borders (Example: 70% train, 10% val, 20% test) ---
        # Using percentages instead of ETT's hardcoded values
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        # Border indices for start of train, val, test sets
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # Border indices for end of train, val, test sets
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        # Select start/end border for the current flag ('train', 'val', 'test')
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # --- Select Numerical Features (Ignoring Date Column, assumed index 0) ---
        if self.features == 'M' or self.features == 'MS':
            # Select all columns EXCEPT the first (date) column
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # Select ONLY the target column. Ensure target is not the date column.
            if self.target == df_raw.columns[0]:
                raise ValueError("Target column cannot be the date column for feature type 'S'.")
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features type: {self.features}")
        

        # df_data.to_csv(r"C:\Users\Hiroshi\OneDrive - Singapore Management University\capstone\MAIN\multivar_time_series_forecasting\datasets\test.csv")
        # --- Find Target Column Index within the selected numerical features ---
        try:
            # Index relative to the columns in df_data
            self.target_col_index = list(df_data.columns).index(self.target)
        except ValueError:
            # If target wasn't found (e.g., using 'S' but target name wrong, or 'M'/'MS' and target name wrong)
             raise ValueError(f"Target column '{self.target}' not found in selected data columns: {list(df_data.columns)}")

        # --- Data Scaling ---
        if self.scale:
            # Fit scaler ONLY on the training portion of the selected numerical data
            train_data_values = df_data.iloc[border1s[0]:border2s[0]].values

            self.scaler.fit(train_data_values)
            # Transform the entire selected numerical data
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values # Use unscaled numerical data

        # --- Store Scaled Numerical Data for Slicing ---
        # We only store the numerical data, sliced according to train/val/test set
        self.data_x = data[border1:border2]
        # data_y is the same sequence, the target column will be sliced out in __getitem__
        self.data_y = data[border1:border2]

        # --- NO Date Processing or data_stamp generation in this class ---

    def __getitem__(self, index):
        # Calculate start/end indices for input sequence (length seq_len)
        s_begin = index
        s_end = s_begin + self.seq_len

        # Calculate start/end indices for the target sequence
        # Note: Dataset_ETT_hour returned shape [label_len+pred_len, num_targets] for seq_y
        # We replicate that length, although label_len isn't used for slicing start here.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Slice the input sequence from numerical data
        seq_x = self.data_x[s_begin:s_end]

        # Slice the target sequence
        # First, get the full feature data for the target timeframe
        seq_y_all_features = self.data_y[r_begin:r_end]
        # Then, select ONLY the target column using the pre-calculated index
        seq_y = seq_y_all_features[:, self.target_col_index:self.target_col_index+1]

        # Return ONLY seq_x and seq_y as FloatTensors
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def __len__(self):
        # Calculate the total number of valid sequences that can start in the current data split
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)



    def inverse_transform(self, data):
        # If no scaling was applied, just pass through
        if not self.scale:
            return data

        scaler = self.scaler
        t_idx = self.target_col_index
        num_features_fit = scaler.n_features_in_

        # Detect type
        is_tensor = isinstance(data, torch.Tensor)
        arr = data.detach().cpu().numpy() if is_tensor else data

        B, L, C_in = arr.shape
        flat = arr.reshape(-1, C_in)  # (B*L, C_in)

        # Determine which column holds the target data
        if C_in == 1:
            # only one channel → assume it's the target
            target_scaled = flat[:, 0]
        else:
            # full set of features → grab the t_idx column
            if t_idx >= C_in:
                raise IndexError(
                    f"target_col_index {t_idx} out of bounds for input with {C_in} channels"
                )
            target_scaled = flat[:, t_idx]

        # Build dummy for full scaler
        dummy = np.zeros((flat.shape[0], num_features_fit), dtype=target_scaled.dtype)
        dummy[:, t_idx] = target_scaled

        # Inverse‐scale and re-extract
        inv_full = scaler.inverse_transform(dummy)  # (B*L, num_features_fit)
        target_unscaled = inv_full[:, t_idx]              # (B*L,)

        # Reshape back to [B, L, 1]
        out = target_unscaled.reshape(B, L, 1)

        # Return same type
        if is_tensor:
            return torch.from_numpy(out).float().to(data.device)
        else:
            return out