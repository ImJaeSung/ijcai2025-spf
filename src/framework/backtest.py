import numpy as np
import pandas as pd

"""Top-K Daily Buy-Hold-Sell Strategy"""

class Backtest():
    def __init__(self, top_k=20):

        self.top_k = top_k
        self.performance = {}
        self.bt_long = 1.0
        self.sharpe_li = []
        self.cumulative_returns = [1.0] # For MDD calculation

    def calculate_mdd(self, cumulative_returns):
        """
        Calculate Maximum Drawdown (MDD) based on the formula:
        MDD = (Trough Value - Peak Value) / Peak Value
        
        Args:
            cumulative_returns (np.array): Array of cumulative returns over time.
            
        Returns:
            float: Maximum Drawdown (MDD) as a positive value.
        """
        # Track the peak value at each time step
        peak_value = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown at each time step
        drawdown = (cumulative_returns - peak_value) / peak_value
        
        # Return the maximum drawdown (in absolute value)
        mdd = np.min(drawdown) # This will be a negative value
        return abs(mdd) # Return the positive magnitude

    def backtesting(self, predictions, gts):
        """
        Evaluate returns, Sharpe ratio, and MDD for a specific phase where both predictions
        and ground truths are in DataFrame format.
        
        Args:
            predictions (pd.DataFrame): Predicted values for all stocks in the phase (index: date, columns: tickers).
            ground_truths (pd.DataFrame): Actual values for all stocks in the phase (index: date, columns: tickers).
            top_k (int): Number of top stocks to consider for evaluation.
            
        Returns:
            dict: Performance metrics including total return, Sharpe ratio, and MDD.
        """
        ground_truth = pd.concat([gts[key]['trend_return'] for key in gts.keys()], axis=1)
        ground_truth.columns = gts.keys()

        # Ensure predictions and ground_truths have the same structure
        assert predictions.shape == ground_truth.shape, "Shape mismatch between predictions and ground_truths"
        assert all(predictions.columns == ground_truth.columns), "Column mismatch between predictions and ground_truths"

        # Iterate over each row (date)
        for date in predictions.index:
            prediction_row = predictions.loc[date].values # Predicted values for the date
            ground_truth_row = ground_truth.loc[date].values # Actual values for the date

            # Top-K ranking for predictions
            rank_pre = np.argsort(prediction_row)
            top_k_indices = rank_pre[-self.top_k:] # Select indices of top-K predictions

            # Backtesting for top-k
            real_ret_rat_top_k = np.sum(ground_truth_row[top_k_indices]) / self.top_k
            self.bt_long += real_ret_rat_top_k
            self.sharpe_li.append(real_ret_rat_top_k)
            self.cumulative_returns.append(self.bt_long)

        # Total return
        self.performance['Return'] = self.bt_long - 1

        # Sharpe ratio
        self.sharpe_li = np.array(self.sharpe_li)
        self.performance['SR'] = (
            np.mean(self.sharpe_li) / np.std(self.sharpe_li)
        ) * 15.87 if len(self.sharpe_li) > 1 else None

        # Maximum Drawdown (MDD)
        self.cumulative_returns = np.array(self.cumulative_returns)
        self.performance['MDD'] = self.calculate_mdd(self.cumulative_returns)

        return self.performance
    

# model_list = [
#     "DTML",
#     "MASTER",
#     # "FactorVAE",
#     "RSR_E",
#     "RSR_I",
#     "RTGCN",
#     "STHANSR",
#     "ESTIMATE",
#     # "THETA",
#     "THETA_sum"
# ]

# phase_dict = {
#     1: {'start' : '2015-05-01', 'end' : '2016-01-01', 'train_start' : '2013-01-01'},
#     2: {'start' : '2016-01-01', 'end' : '2016-09-01', 'train_start' : '2013-09-01'},
#     3: {'start' : '2016-09-01', 'end' : '2017-05-01', 'train_start' : '2014-05-01'},
#     4: {'start' : '2017-05-01', 'end' : '2018-01-01', 'train_start' : '2015-01-01'},
#     5: {'start' : '2018-01-01', 'end' : '2018-09-01', 'train_start' : '2015-09-01'},
#     6: {'start' : '2018-09-01', 'end' : '2019-05-01', 'train_start' : '2016-05-01'},
#     7: {'start' : '2019-05-01', 'end' : '2020-01-01', 'train_start' : '2017-01-01'},
#     8: {'start' : '2020-01-01', 'end' : '2020-09-01', 'train_start' : '2017-09-01'},
#     9: {'start' : '2020-09-01', 'end' : '2021-05-01', 'train_start' : '2018-05-01'},
#     10: {'start' : '2021-05-01', 'end' : '2022-01-01', 'train_start' : '2019-01-01'}
# }

# phase_list = [f"Phase {idx}" for idx in range(1,11)]
# columns = phase_list + ["Total"]

# return_df = pd.DataFrame(index=model_list, columns=columns)
# sr_df = pd.DataFrame(index=model_list, columns=columns)
# mdd_df = pd.DataFrame(index=model_list, columns=columns)

# for model_name in tqdm(model_list):

#     pred_df = pd.read_csv(f"./Prediction/{model_name}.csv", index_col=0)
        
#     cumulative_returns_phase_dict = {}

#     for i in range(10):
#         phase, date = list(phase_dict.items())[i]
#         pred_df = pd.read_csv(f"./Prediction/{model_name}.csv", index_col=0)
                        
#         pred_df_phase = pred_df[(pred_df.index >= date['start']) & (pred_df.index < date['end'])]
#         with open(f"../Data/sp500/{date['train_start']}/test_gts.pkl", "rb") as f:
#             gts = pickle.load(f)
#         gts_df_phase = pd.concat([gts[key]['trend_return'] for key in gts.keys()], axis=1)
#         gts_df_phase.columns = gts.keys()

#         bt = Backtest()
#         performance = bt.backtesting(pred_df_phase, gts_df_phase)

#         return_df.loc[model_name, f"Phase {phase}"] = performance['Return']
#         sr_df.loc[model_name, f"Phase {phase}"] = performance['SR']
#         mdd_df.loc[model_name, f"Phase {phase}"] = performance['MDD']

# return_df['Total'] = return_df.iloc[:, :-1].mean(axis=1)
# sr_df['Total'] = sr_df.iloc[:, :-1].mean(axis=1)
# mdd_df['Total'] = mdd_df.iloc[:, :-1].mean(axis=1)

# return_df = return_df.astype(float).round(3)
# sr_df = sr_df.astype(float).round(3)
# mdd_df = mdd_df.astype(float).round(3)

# total_df = pd.concat([return_df['Total'], sr_df['Total'], mdd_df['Total']], axis=1)
# total_df.columns = ["Return", "SR", "MDD"]