import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime
import os

class MLBacktester:
    """
    A backtesting library for quantitative trading with ML integration
    to optimize signal combinations with support for training on dataset A
    and finding top combinations + backtesting on dataset B.
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.signals = None
        self.price_column = 'close'  # Default price column
        self.model = None
        self.signal_combinations = None
        self.top_combinations = None
        self.backtest_results = {}
        self.exclude_columns = ['datetime', 'start_time', 'close', 'high', 'low', 'open', 'volume']
        
    def load_training_data(self, filepath, price_column='close'):
        """
        Load training data from a CSV file with 'datetime' and multiple signal columns.
        """
        self.train_data = pd.read_csv(filepath)
        self.price_column = price_column
        
        # Ensure datetime is properly converted
        self.train_data['datetime'] = pd.to_datetime(self.train_data['datetime'])
        self.train_data.set_index('datetime', inplace=True)
    
        # Extract potential signal columns (excluding specific columns)
        all_columns = list(self.train_data.columns)
        self.signals = [col for col in all_columns if col not in self.exclude_columns]
    
        print(f"Loaded training data: {len(self.train_data)} rows with {len(self.signals)} potential signals.")
        return self

    def load_testing_data(self, filepath):
        """
        Load testing/backtesting data from a CSV file with 'datetime' and signal columns 
        that match the training data.
        """
        self.test_data = pd.read_csv(filepath)
        self.test_data['datetime'] = pd.to_datetime(self.test_data['datetime'])
        self.test_data.set_index('datetime', inplace=True)
    
        # Verify that testing data has the same signals as training data
        for signal in self.signals:
            if signal not in self.test_data.columns:
                raise ValueError(f"Signal '{signal}' from training data not found in testing data")
    
        print(f"Loaded testing data: {len(self.test_data)} rows.")
        return self
        
    def _prepare_training_data(self, lookback=20, forward_return_days=5, return_threshold=0.01, 
                              indicator_columns=None, explicit_signals=None):
        """
        Prepare training data for the ML model using signals from the training dataset.
        """
        if self.train_data is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")
    
        # Make sure we have enough data for the required lookback and forward periods
        if len(self.train_data) <= lookback + forward_return_days:
            raise ValueError(f"Not enough data points. Need more than {lookback + forward_return_days} rows.")
    
        # Calculate forward returns for target creation
        self.train_data['forward_return'] = self.train_data[self.price_column].pct_change(
            periods=forward_return_days).shift(-forward_return_days)
        
        # Determine which signals/indicators to use
        if explicit_signals:
            # Use explicitly provided signal columns
            use_signals = [s for s in explicit_signals if s in self.signals]
        elif indicator_columns:
            # Use specified indicator columns
            use_signals = [s for s in indicator_columns if s in self.signals]
        else:
            # Default: use columns that end with "_signal" or specific indicator types
            signal_columns = [col for col in self.signals if col.endswith('_signal')]
            indicator_columns = [col for col in self.signals if any(col.startswith(prefix) 
                                for prefix in ['MA_', 'EMA_', 'RSI_', 'MACD', 'Bollinger_'])]
            use_signals = signal_columns + indicator_columns
        
        # Make sure we have enough signals
        if len(use_signals) < 2:
            raise ValueError(f"Not enough valid signals found. Need at least 2, found {len(use_signals)}.")
            
        print(f"Using {len(use_signals)} signals/indicators: {use_signals}")
        
        # Create combinations of signals
        combs = list(combinations(use_signals, 2))
        print(f"Working with {len(combs)} signal combinations")
    
        # Prepare feature matrix
        features_list = []
    
        # For each valid data point
        for i in range(lookback, len(self.train_data) - forward_return_days):
            row_features = []
        
            # For each signal combination
            for sig1, sig2 in combs:
                # Get signal values in the lookback window
                window1 = self.train_data[sig1].iloc[i-lookback:i]
                window2 = self.train_data[sig2].iloc[i-lookback:i]
            
                # Check for NaN values
                if window1.isnull().any() or window2.isnull().any():
                    # Use default values if we have NaNs
                    row_features.extend([0, 0, 1])
                    continue
            
                # Calculate features for this combination
                correlation = window1.corr(window2)
                mean_diff = (window1 - window2).mean()
                std_ratio = window1.std() / (window2.std() + 1e-10)  # Avoid division by zero
            
                row_features.extend([correlation, mean_diff, std_ratio])
        
            features_list.append(row_features)
    
        # Convert to numpy array
        X = np.array(features_list)
    
        # Print shape to debug
        print(f"Feature matrix shape: {X.shape}")
    
        # Create target labels based on forward returns
        y = (self.train_data['forward_return'].iloc[lookback:-forward_return_days] > return_threshold).astype(int)
        print(f"Target vector shape: {y.shape}")
    
        return X, y, combs

    def _prepare_testing_data(self, lookback=20, forward_return_days=5, return_threshold=0.01):
        """
        Prepare testing data using the same signal combinations as in training.
        """
        if self.test_data is None:
            raise ValueError("Testing data not loaded. Call load_testing_data() first.")
    
        if self.signal_combinations is None:
            raise ValueError("No signal combinations available. Train the model first.")
    
        # Calculate forward returns
        self.test_data['forward_return'] = self.test_data[self.price_column].pct_change(
            periods=forward_return_days).shift(-forward_return_days)
    
        # Use the same combinations as in training
        combs = self.signal_combinations
    
        # Prepare features
        features_list = []
    
        for i in range(lookback, len(self.test_data) - forward_return_days):
            row_features = []
        
            for sig1, sig2 in combs:
                window1 = self.test_data[sig1].iloc[i-lookback:i]
                window2 = self.test_data[sig2].iloc[i-lookback:i]
            
                if window1.isnull().any() or window2.isnull().any():
                    row_features.extend([0, 0, 1])
                    continue
                
                correlation = window1.corr(window2)
                mean_diff = (window1 - window2).mean()
                std_ratio = window1.std() / (window2.std() + 1e-10)
            
                row_features.extend([correlation, mean_diff, std_ratio])
        
            features_list.append(row_features)
    
        X = np.array(features_list)
        print(f"Test feature matrix shape: {X.shape}")
    
        # Create target labels
        y = (self.test_data['forward_return'].iloc[lookback:-forward_return_days] > return_threshold).astype(int)
        print(f"Test target vector shape: {y.shape}")
    
        return X, y, combs

    
    def train_model(self, lookback=20, forward_return_days=5, return_threshold=0.01, 
                    indicator_columns=None, explicit_signals=None):
        """
        Train the XGBoost model on the training dataset to identify the best signal combinations.
        
        Parameters:
        -----------
        lookback : int
            Number of periods to look back for feature calculations
        forward_return_days : int
            Number of days forward to calculate returns for the target
        return_threshold : float
            Threshold for positive return classification
        indicator_columns : list, optional
            List of indicator columns to use. If None, will use heuristics.
        explicit_signals : list, optional
            Explicitly specify signal columns to use
        """
        # Prepare the training data
        X, y, combs = self._prepare_training_data(
            lookback, forward_return_days, return_threshold, 
            indicator_columns, explicit_signals)
    
        # Initialize and train the XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
        # Train the model on the prepared data
        self.model.fit(X, y)
    
        # Evaluate the model on the training data
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Model trained on training dataset with accuracy: {accuracy:.2f}")
    
        # Store the signal combinations for later use
        self.signal_combinations = combs
    
        return self
    
    def find_top_combinations(self, n=5, lookback=20, forward_return_days=5, return_threshold=0.01):
        """
        Use the trained model to find the top n signal combinations based on test dataset performance.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.test_data is None:
            raise ValueError("Testing data not loaded. Call load_testing_data() first.")
        
        # Prepare test data to evaluate combinations
        X_test, y_test, combs = self._prepare_testing_data(
            lookback, forward_return_days, return_threshold)
        
        # Get predictions on test data
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Model performance on test dataset: accuracy = {test_accuracy:.2f}")
        
        # Calculate performance for each combination on the test data
        features_per_comb = 3  # correlation, mean_diff, std_ratio
        comb_performances = []
        
        for i in range(len(self.signal_combinations)):
            # Get feature indices for this combination
            start_idx = i * features_per_comb
            end_idx = start_idx + features_per_comb
            
            # Create a model with just these features
            X_combo = X_test[:, start_idx:end_idx]
            
            # Simple performance measure: correlation with target
            combo_performance = np.abs(np.corrcoef(X_combo.mean(axis=1), y_test)[0, 1])
            if np.isnan(combo_performance):
                combo_performance = 0
                
            comb_performances.append((self.signal_combinations[i], combo_performance))
        
        # Sort combinations by performance and get top n
        comb_performances.sort(key=lambda x: x[1], reverse=True)
        self.top_combinations = comb_performances[:n]
        
        print(f"Top {n} signal combinations based on test dataset performance:")
        for i, (comb, performance) in enumerate(self.top_combinations, 1):
            print(f"{i}. {comb[0]} + {comb[1]} (Performance Score: {performance:.4f})")
        
        return self
    
    def _generate_trading_signals(self, combination, lookback=5):
        """
        Generate trading signals based on a combination of two signals using the test dataset.
        """
        if self.test_data is None:
            raise ValueError("Testing data not loaded. Call load_testing_data() first.")
            
        sig1, sig2 = combination
        signals = pd.DataFrame(index=self.test_data.index)
        signals['signal'] = 0
        
        # Simple example: Generate buy signal when sig1 > sig2 over lookback period
        for i in range(lookback, len(self.test_data)):
            window1 = self.test_data[sig1].iloc[i-lookback:i]
            window2 = self.test_data[sig2].iloc[i-lookback:i]
            
            if window1.mean() > window2.mean():
                signals.iloc[i, 0] = 1  # Buy signal
            elif window1.mean() < window2.mean():
                signals.iloc[i, 0] = -1  # Sell signal
        
        return signals
    
    def backtest(self, initial_capital=100000, transaction_cost=0.001):
        """
        Run backtests for the top signal combinations on the testing dataset.
        """
        if self.top_combinations is None:
            raise ValueError("No top combinations found. Call find_top_combinations() first.")
        
        if self.test_data is None:
            raise ValueError("Testing data not loaded. Call load_testing_data() first.")
        
        for i, (combination, _) in enumerate(self.top_combinations, 1):
            signals = self._generate_trading_signals(combination)
            
            # Initialize portfolio
            portfolio = pd.DataFrame(index=signals.index)
            portfolio['signal'] = signals['signal']
            portfolio['return'] = self.test_data[self.price_column].pct_change()  # Using price column
            
            # Calculate position and portfolio value
            portfolio['position'] = portfolio['signal'].shift(1)
            portfolio['strategy_return'] = portfolio['position'] * portfolio['return']
            
            # Apply transaction costs
            portfolio['trades'] = portfolio['position'].diff().abs()
            portfolio['transaction_cost'] = portfolio['trades'] * transaction_cost
            portfolio['strategy_return'] = portfolio['strategy_return'] - portfolio['transaction_cost']
            
            # Calculate cumulative returns
            portfolio['cum_return'] = (1 + portfolio['strategy_return']).fillna(1).cumprod()
            portfolio['cum_strategy_value'] = initial_capital * portfolio['cum_return']
            
            # Calculate metrics
            total_return = portfolio['cum_return'].iloc[-1] - 1
            sharpe_ratio = portfolio['strategy_return'].mean() / (portfolio['strategy_return'].std() + 1e-10) * np.sqrt(252)
            
            # Calculate drawdowns
            portfolio['peak'] = portfolio['cum_strategy_value'].cummax()
            portfolio['drawdown'] = (portfolio['cum_strategy_value'] - portfolio['peak']) / portfolio['peak']
            max_drawdown = portfolio['drawdown'].min()
            
            # Calculate trade metrics
            trades_count = portfolio['trades'].sum() / 2  # Divide by 2 as each round trip is counted twice
            win_count = len(portfolio[portfolio['strategy_return'] > 0])
            loss_count = len(portfolio[portfolio['strategy_return'] < 0])
            win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
            
            # Store results
            combination_name = f"{combination[0]}_{combination[1]}"
            self.backtest_results[combination_name] = {
                'portfolio': portfolio,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades_count': trades_count,
                'win_rate': win_rate
            }
            
            print(f"Backtest {i} completed: {combination_name}")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            print(f"  Trades: {trades_count}")
            print(f"  Win Rate: {win_rate:.2%}")
            print("-" * 40)
        
        return self
    
    def plot_results(self, figsize=(12, 8)):
        """
        Plot backtest results for comparison.
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        plt.figure(figsize=figsize)
        
        for name, result in self.backtest_results.items():
            plt.plot(result['portfolio']['cum_strategy_value'], label=name)
        
        plt.title('Backtest Results Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.png"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.show()
        
        return self
    
    def export_results(self, directory=None):
        """
        Export backtest results to CSV files.
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        if directory is None:
            directory = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(directory, exist_ok=True)
        
        # Export summary metrics
        summary = []
        for name, result in self.backtest_results.items():
            summary.append({
                'Combination': name,
                'Total Return': result['total_return'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Trades Count': result['trades_count'],
                'Win Rate': result['win_rate']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{directory}/summary_metrics.csv", index=False)
        
        # Export detailed results for each combination
        for name, result in self.backtest_results.items():
            result['portfolio'].to_csv(f"{directory}/{name}_detailed.csv")
        
        print(f"Results exported to directory: {directory}")
        return self


# Example usage
#if __name__ == "__main__":
    # Example code to demonstrate how to use the library
    #backtester = MLBacktester()
    
    # Load training data (dataset A)
    #backtester.load_training_data("training_signals.csv", price_column='close')
    
    # Train the model on dataset A with explicit signals
    # You can specify which signals to use:
    # 1. Use signals ending with '_signal' and common indicators
    #backtester.train_model(lookback=20, forward_return_days=5)
    
    # 2. Or explicitly specify which columns to use as signals
    # specific_signals = ['rule1_signal', 'rule2_signal', 'RSI_14', 'MACD', 'EMA_12', 'EMA_26']
    # backtester.train_model(lookback=20, forward_return_days=5, explicit_signals=specific_signals)
    
    # Load testing data (dataset B)
    #backtester.load_testing_data("testing_signals.csv")
    
    # Find top combinations based on testing dataset B performance
    #backtester.find_top_combinations(n=5)
    
    # Run backtests on dataset B using the top combinations identified
    #backtester.backtest(initial_capital=100000)
    
    # Visualize results
    #backtester.plot_results()
    
    # Export results
    #backtester.export_results()