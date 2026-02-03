# continuous_learning.py
"""
Continuous Learning System for Trading Bot
- Logs all trades and outcomes
- Monitors performance degradation
- Auto-retrains model when performance drops
- Validates new models before deployment
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import joblib
import threading
import time

# Config imports
from config import (
    API_KEY, API_SECRET, SYMBOL, INTERVALS, BEST_MODEL_PATH,
    SEQ_LEN_MODEL, ENTRY_THRESHOLD, TP_MIN, SL_MIN, FUTURE_WINDOW
)

# Import continuous learning config if available
try:
    from config import (
        CL_MIN_WIN_RATE, CL_MIN_PROFIT_FACTOR, CL_PERFORMANCE_WINDOW,
        CL_RETRAIN_INTERVAL_DAYS, CL_RETRAIN_DATA_DAYS, CL_MIN_TRADES_FOR_EVAL,
        CL_CHECK_INTERVAL_HOURS, CL_AUTO_RETRAIN,
        CL_FINE_TUNE_LR, CL_FINE_TUNE_EPOCHS, CL_FINE_TUNE_BATCH_SIZE
    )
    MIN_WIN_RATE = CL_MIN_WIN_RATE
    MIN_PROFIT_FACTOR = CL_MIN_PROFIT_FACTOR
    PERFORMANCE_WINDOW = CL_PERFORMANCE_WINDOW
    RETRAIN_INTERVAL_DAYS = CL_RETRAIN_INTERVAL_DAYS
    RETRAIN_DATA_DAYS = CL_RETRAIN_DATA_DAYS
    MIN_TRADES_FOR_EVAL = CL_MIN_TRADES_FOR_EVAL
    DEGRADATION_CHECK_INTERVAL = CL_CHECK_INTERVAL_HOURS * 3600
    FINE_TUNE_LR = CL_FINE_TUNE_LR
    FINE_TUNE_EPOCHS = CL_FINE_TUNE_EPOCHS
    FINE_TUNE_BATCH_SIZE = CL_FINE_TUNE_BATCH_SIZE
except ImportError:
    # Fallback defaults
    MIN_WIN_RATE = 0.65
    MIN_PROFIT_FACTOR = 1.2
    PERFORMANCE_WINDOW = 50
    RETRAIN_INTERVAL_DAYS = 7
    RETRAIN_DATA_DAYS = 30
    MIN_TRADES_FOR_EVAL = 20
    DEGRADATION_CHECK_INTERVAL = 3600
    FINE_TUNE_LR = 0.0001
    FINE_TUNE_EPOCHS = 30
    FINE_TUNE_BATCH_SIZE = 32

# ============================================================
# CONFIGURATION (from config.py or defaults)
# ============================================================

VALIDATION_TRADES = 30                 # Backtest new model on last 30 trades

# File paths
TRADE_LOG_PATH = "trade_log.json"
PERFORMANCE_LOG_PATH = "performance_log.json"
MODEL_HISTORY_PATH = "model_history.json"
RETRAIN_LOCK_PATH = "retrain.lock"


# ============================================================
# TRADE LOGGER
# ============================================================

class TradeLogger:
    """
    Logs all trades with entry, exit, and outcome data.
    Used for performance tracking and model retraining.
    """

    def __init__(self, log_path=TRADE_LOG_PATH):
        self.log_path = log_path
        self.trades = self._load_trades()

    def _load_trades(self):
        """Load existing trades from file."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load trade log: {e}")
        return []

    def _save_trades(self):
        """Save trades to file."""
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save trade log: {e}")

    def log_entry(self, trade_id, symbol, side, entry_price, qty,
                  confidence, model_version, features=None):
        """Log trade entry."""
        trade = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'qty': qty,
            'confidence': confidence,
            'model_version': model_version,
            'entry_time': datetime.now().isoformat(),
            'status': 'OPEN',
            'exit_price': None,
            'exit_time': None,
            'pnl': None,
            'pnl_percent': None,
            'exit_reason': None,
            'features_snapshot': features
        }
        self.trades.append(trade)
        self._save_trades()
        print(f"[TRADE-LOG] Logged entry: {side} {symbol} @ ${entry_price:.2f}")
        return trade_id

    def log_exit(self, trade_id, exit_price, exit_reason='UNKNOWN'):
        """Log trade exit and calculate PnL."""
        for trade in self.trades:
            if trade['trade_id'] == trade_id and trade['status'] == 'OPEN':
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_reason'] = exit_reason
                trade['status'] = 'CLOSED'

                # Calculate PnL
                entry = trade['entry_price']
                if trade['side'] == 'LONG':
                    trade['pnl_percent'] = (exit_price - entry) / entry * 100
                else:  # SHORT
                    trade['pnl_percent'] = (entry - exit_price) / entry * 100

                trade['pnl'] = trade['pnl_percent'] * trade['qty'] * entry / 100

                self._save_trades()

                result = 'WIN' if trade['pnl_percent'] > 0 else 'LOSS'
                print(f"[TRADE-LOG] Logged exit: {result} {trade['pnl_percent']:.2f}% (${trade['pnl']:.2f})")
                return trade

        print(f"[WARN] Trade {trade_id} not found or already closed")
        return None

    def get_recent_trades(self, n=50, closed_only=True):
        """Get last N trades."""
        if closed_only:
            closed = [t for t in self.trades if t['status'] == 'CLOSED']
        else:
            closed = self.trades
        return closed[-n:] if len(closed) >= n else closed

    def get_trades_since(self, since_date):
        """Get trades since a specific date."""
        trades = []
        for t in self.trades:
            if t['status'] == 'CLOSED':
                exit_time = datetime.fromisoformat(t['exit_time'])
                if exit_time >= since_date:
                    trades.append(t)
        return trades

    def get_open_trades(self):
        """Get all open trades."""
        return [t for t in self.trades if t['status'] == 'OPEN']


# ============================================================
# PERFORMANCE MONITOR
# ============================================================

class PerformanceMonitor:
    """
    Monitors trading performance and detects degradation.
    Triggers retraining when performance drops below thresholds.
    """

    def __init__(self, trade_logger, log_path=PERFORMANCE_LOG_PATH):
        self.trade_logger = trade_logger
        self.log_path = log_path
        self.performance_history = self._load_history()
        self.alerts = deque(maxlen=100)

    def _load_history(self):
        """Load performance history."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save_history(self):
        """Save performance history."""
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save performance log: {e}")

    def calculate_metrics(self, trades):
        """Calculate performance metrics from trades."""
        if not trades:
            return None

        wins = sum(1 for t in trades if t['pnl_percent'] > 0)
        losses = sum(1 for t in trades if t['pnl_percent'] <= 0)
        total = wins + losses

        if total == 0:
            return None

        win_rate = wins / total

        # Profit factor
        total_wins = sum(t['pnl_percent'] for t in trades if t['pnl_percent'] > 0)
        total_losses = abs(sum(t['pnl_percent'] for t in trades if t['pnl_percent'] <= 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average trade
        avg_trade = np.mean([t['pnl_percent'] for t in trades])

        # Max drawdown
        cumulative = []
        running = 0
        for t in trades:
            running += t['pnl_percent']
            cumulative.append(running)

        peak = 0
        max_dd = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        # Sharpe-like ratio
        returns = [t['pnl_percent'] for t in trades]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Average confidence of model predictions
        avg_confidence = np.mean([t['confidence'] for t in trades if t['confidence']])

        return {
            'timestamp': datetime.now().isoformat(),
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'avg_confidence': avg_confidence,
            'total_pnl': sum(t['pnl_percent'] for t in trades)
        }

    def check_performance(self):
        """
        Check current performance and return (is_degraded, metrics, reasons).
        """
        trades = self.trade_logger.get_recent_trades(PERFORMANCE_WINDOW)

        if len(trades) < MIN_TRADES_FOR_EVAL:
            return False, None, [f"Not enough trades ({len(trades)}/{MIN_TRADES_FOR_EVAL})"]

        metrics = self.calculate_metrics(trades)
        if not metrics:
            return False, None, ["Could not calculate metrics"]

        # Log metrics
        self.performance_history.append(metrics)
        self._save_history()

        # Check for degradation
        reasons = []
        is_degraded = False

        if metrics['win_rate'] < MIN_WIN_RATE:
            reasons.append(f"Win rate {metrics['win_rate']:.1%} < {MIN_WIN_RATE:.1%}")
            is_degraded = True

        if metrics['profit_factor'] < MIN_PROFIT_FACTOR:
            reasons.append(f"Profit factor {metrics['profit_factor']:.2f} < {MIN_PROFIT_FACTOR}")
            is_degraded = True

        # Check trend (last 3 checks)
        if len(self.performance_history) >= 3:
            recent_wr = [h['win_rate'] for h in self.performance_history[-3:]]
            if all(recent_wr[i] > recent_wr[i+1] for i in range(len(recent_wr)-1)):
                reasons.append("Win rate declining for 3 consecutive checks")
                is_degraded = True

        return is_degraded, metrics, reasons

    def get_performance_summary(self):
        """Get current performance summary."""
        trades = self.trade_logger.get_recent_trades(PERFORMANCE_WINDOW)
        metrics = self.calculate_metrics(trades)

        if not metrics:
            return "Not enough data for performance summary"

        summary = f"""
=== PERFORMANCE SUMMARY ===
Trades: {metrics['total_trades']} (W:{metrics['wins']} / L:{metrics['losses']})
Win Rate: {metrics['win_rate']:.1%}
Profit Factor: {metrics['profit_factor']:.2f}
Avg Trade: {metrics['avg_trade']:.2f}%
Total PnL: {metrics['total_pnl']:.2f}%
Max Drawdown: {metrics['max_drawdown']:.2f}%
Sharpe: {metrics['sharpe']:.2f}
Avg Confidence: {metrics['avg_confidence']:.2f}
===========================
"""
        return summary


# ============================================================
# AUTO RETRAINER
# ============================================================

class AutoRetrainer:
    """
    Automatically retrains model when performance degrades.
    Validates new model before deployment.
    """

    def __init__(self, trade_logger, performance_monitor):
        self.trade_logger = trade_logger
        self.performance_monitor = performance_monitor
        self.model_history = self._load_model_history()
        self.is_retraining = False
        self.last_retrain = self._get_last_retrain_time()

    def _load_model_history(self):
        """Load model version history."""
        if os.path.exists(MODEL_HISTORY_PATH):
            try:
                with open(MODEL_HISTORY_PATH, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save_model_history(self):
        """Save model version history."""
        try:
            with open(MODEL_HISTORY_PATH, 'w') as f:
                json.dump(self.model_history, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save model history: {e}")

    def _get_last_retrain_time(self):
        """Get timestamp of last retraining."""
        if self.model_history:
            last = self.model_history[-1]
            return datetime.fromisoformat(last['timestamp'])
        return datetime.now() - timedelta(days=RETRAIN_INTERVAL_DAYS + 1)

    def _can_retrain(self):
        """Check if enough time has passed since last retrain."""
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain >= timedelta(days=RETRAIN_INTERVAL_DAYS)

    def _fetch_recent_data(self, days=RETRAIN_DATA_DAYS):
        """Fetch recent market data for retraining."""
        from feature_pipeline import fetch_features_multi_timeframe

        print(f"[RETRAIN] Fetching last {days} days of data...")
        df = fetch_features_multi_timeframe()

        # Filter to recent data
        # Assuming data is sorted by time, take last N rows
        # Each row is 1 candle of primary timeframe
        candles_per_day = 24 if '1h' in INTERVALS[0] else 96  # 1h or 15m
        rows_needed = days * candles_per_day

        if len(df) > rows_needed:
            df = df.tail(rows_needed)

        return df

    def _fine_tune_model(self, df):
        """
        Fine-tune existing model on recent data.

        Fine-tuning advantages over full retraining:
        - Preserves learned patterns from original training
        - Faster (fewer epochs needed)
        - Uses recent data to adapt to market changes
        - Lower risk of catastrophic forgetting
        """
        from sklearn.preprocessing import RobustScaler
        from tensorflow.keras.utils import to_categorical
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras.models import load_model
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf

        print("[FINE-TUNE] Fine-tuning existing model on recent data...")

        # Import labeling function
        try:
            from train_lstm_keras import triple_barrier_labels, FocalLoss
        except ImportError:
            print("[ERROR] Could not import training functions")
            return None

        # Load existing model
        try:
            # Register custom loss function
            custom_objects = {'FocalLoss': FocalLoss}
            model = load_model(BEST_MODEL_PATH, custom_objects=custom_objects)
            print(f"[FINE-TUNE] Loaded existing model: {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            return None

        # Load existing scaler or create new one
        try:
            scaler = joblib.load("scaler.pkl")
            X = scaler.transform(df)
            print("[FINE-TUNE] Using existing scaler")
        except:
            print("[FINE-TUNE] Creating new scaler")
            scaler = RobustScaler()
            X = scaler.fit_transform(df)

        close = df[f'close_{INTERVALS[0]}'].values
        high = df[f'high_{INTERVALS[0]}'].values
        low = df[f'low_{INTERVALS[0]}'].values
        atr = df[f'atr_{INTERVALS[0]}'].values

        # Generate labels using same method as original training
        labels = triple_barrier_labels(
            close, high, low,
            tp_mult=1.5,
            sl_mult=1.0,
            max_holding=FUTURE_WINDOW,
            atr=atr
        )

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(SEQ_LEN_MODEL, len(X) - FUTURE_WINDOW):
            X_seq.append(X[i - SEQ_LEN_MODEL:i])
            y_seq.append(labels[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        if len(X_seq) < 100:
            print(f"[WARN] Not enough data for fine-tuning: {len(X_seq)} samples")
            return None

        # Split (80/20)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train_raw, y_val_raw = y_seq[:split], y_seq[split:]

        # Class weights
        classes = np.unique(y_train_raw)
        if len(classes) < 2:
            print("[WARN] Not enough class diversity for fine-tuning")
            return None

        weights = compute_class_weight("balanced", classes=classes, y=y_train_raw)
        class_weights = dict(zip(classes.astype(int), weights))

        # One-hot encode
        y_train = to_categorical(y_train_raw, num_classes=3)
        y_val = to_categorical(y_val_raw, num_classes=3)

        # === FINE-TUNING STRATEGY ===
        # 1. Use lower learning rate (10x smaller than initial training)
        # 2. Fewer epochs (model already knows patterns)
        # 3. Early stopping to prevent overfitting

        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss=model.loss,
            metrics=['accuracy']
        )

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        print(f"[FINE-TUNE] Training with LR={FINE_TUNE_LR}, epochs={FINE_TUNE_EPOCHS}, batch={FINE_TUNE_BATCH_SIZE}")
        print(f"[FINE-TUNE] Data: {len(X_train)} train, {len(X_val)} validation samples")

        history = model.fit(
            X_train, y_train,
            epochs=FINE_TUNE_EPOCHS,
            batch_size=FINE_TUNE_BATCH_SIZE,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.001  # Only stop if no improvement
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ],
            verbose=1
        )

        # Save scaler (in case it was updated)
        joblib.dump(scaler, "scaler_new.pkl")

        # Report improvement
        initial_loss = history.history['val_loss'][0]
        final_loss = history.history['val_loss'][-1]
        initial_acc = history.history['val_accuracy'][0]
        final_acc = history.history['val_accuracy'][-1]

        print(f"[FINE-TUNE] Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"[FINE-TUNE] Accuracy: {initial_acc:.4f} -> {final_acc:.4f}")

        return model, history, X_val, y_val_raw

    def _train_new_model(self, df):
        """Wrapper that uses fine-tuning instead of full retraining."""
        return self._fine_tune_model(df)

    def _validate_model(self, new_model, X_val, y_val):
        """Validate new model against recent trades."""
        print("[RETRAIN] Validating new model...")

        # Get predictions
        y_pred = new_model.predict(X_val, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_pred_conf = np.max(y_pred, axis=1)

        # Filter high confidence predictions
        mask = (y_pred_conf >= ENTRY_THRESHOLD) & (y_pred_class != 0)

        if mask.sum() < 10:
            print("[WARN] Not enough high-confidence predictions for validation")
            return None

        # Calculate win rate on filtered predictions
        correct = (y_pred_class[mask] == y_val[mask]).sum()
        total = mask.sum()
        accuracy = correct / total

        # Calculate precision for LONG/SHORT
        long_mask = mask & (y_pred_class == 1)
        short_mask = mask & (y_pred_class == 2)

        long_precision = (y_val[long_mask] == 1).sum() / long_mask.sum() if long_mask.sum() > 0 else 0
        short_precision = (y_val[short_mask] == 2).sum() / short_mask.sum() if short_mask.sum() > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'long_precision': long_precision,
            'short_precision': short_precision,
            'total_signals': total,
            'avg_confidence': y_pred_conf[mask].mean()
        }

        print(f"[RETRAIN] Validation results:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  LONG precision: {long_precision:.1%}")
        print(f"  SHORT precision: {short_precision:.1%}")
        print(f"  Signals: {total}")

        return metrics

    def _compare_with_current(self, new_metrics):
        """Compare new model with current model performance."""
        current_metrics = self.performance_monitor.calculate_metrics(
            self.trade_logger.get_recent_trades(VALIDATION_TRADES)
        )

        if not current_metrics:
            print("[WARN] No current metrics available, accepting new model")
            return True

        print(f"[RETRAIN] Comparison:")
        print(f"  Current win rate: {current_metrics['win_rate']:.1%}")
        print(f"  New model accuracy: {new_metrics['accuracy']:.1%}")

        # New model should be better
        # Use weighted average of precision metrics
        new_score = (new_metrics['long_precision'] + new_metrics['short_precision']) / 2
        current_score = current_metrics['win_rate']

        improvement = new_score - current_score
        print(f"  Improvement: {improvement:+.1%}")

        # Accept if improvement > 0 or current is very bad
        return improvement > -0.05 or current_metrics['win_rate'] < 0.55

    def fine_tune(self, force=False):
        """
        Execute fine-tuning process.
        Returns (success, message, new_model_path)
        """
        if self.is_retraining:
            return False, "Fine-tuning already in progress", None

        if not force and not self._can_retrain():
            days_left = RETRAIN_INTERVAL_DAYS - (datetime.now() - self.last_retrain).days
            return False, f"Too soon to fine-tune. Wait {days_left} more days.", None

        self.is_retraining = True

        try:
            # Create lock file
            with open(RETRAIN_LOCK_PATH, 'w') as f:
                f.write(datetime.now().isoformat())

            # Step 1: Fetch data
            df = self._fetch_recent_data()
            if df is None or len(df) < SEQ_LEN_MODEL * 2:
                return False, "Not enough data for fine-tuning", None

            # Step 2: Fine-tune model
            result = self._fine_tune_model(df)
            if result is None:
                return False, "Fine-tuning failed", None

            new_model, history, X_val, y_val = result

            # Step 3: Validate
            new_metrics = self._validate_model(new_model, X_val, y_val)
            if new_metrics is None:
                return False, "Validation failed", None

            # Step 4: Compare with current
            should_deploy = self._compare_with_current(new_metrics)

            if should_deploy:
                # Save new model
                version = f"ft{len(self.model_history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                new_model_path = f"model_{version}.keras"
                new_model.save(new_model_path)

                # Backup current model
                if os.path.exists(BEST_MODEL_PATH):
                    backup_path = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.keras"
                    import shutil
                    shutil.copy2(BEST_MODEL_PATH, backup_path)
                    print(f"[FINE-TUNE] Backup saved: {backup_path}")

                # Deploy fine-tuned model
                new_model.save(BEST_MODEL_PATH)
                if os.path.exists("scaler_new.pkl"):
                    import shutil
                    shutil.copy2("scaler_new.pkl", "scaler.pkl")

                # Log to history
                self.model_history.append({
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'fine-tune',
                    'metrics': new_metrics,
                    'path': new_model_path,
                    'deployed': True
                })
                self._save_model_history()

                self.last_retrain = datetime.now()

                print(f"[FINE-TUNE] SUCCESS! Fine-tuned model deployed: {version}")
                return True, f"Fine-tuned model {version} deployed successfully", new_model_path
            else:
                print("[FINE-TUNE] Fine-tuned model not better than current. Keeping existing model.")
                return False, "Fine-tuned model not better than current", None

        except Exception as e:
            print(f"[ERROR] Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Error: {str(e)}", None

        finally:
            self.is_retraining = False
            if os.path.exists(RETRAIN_LOCK_PATH):
                os.remove(RETRAIN_LOCK_PATH)

    def retrain(self, force=False):
        """Alias for fine_tune for backward compatibility."""
        return self.fine_tune(force)


# ============================================================
# CONTINUOUS LEARNING MANAGER
# ============================================================

class ContinuousLearningManager:
    """
    Main manager that coordinates logging, monitoring, and retraining.
    Run this in a background thread for autonomous operation.
    """

    def __init__(self):
        self.trade_logger = TradeLogger()
        self.performance_monitor = PerformanceMonitor(self.trade_logger)
        self.auto_retrainer = AutoRetrainer(self.trade_logger, self.performance_monitor)
        self.running = False
        self.monitor_thread = None

    def log_trade_entry(self, trade_id, symbol, side, entry_price, qty,
                        confidence, model_version='unknown', features=None):
        """Log a new trade entry."""
        return self.trade_logger.log_entry(
            trade_id, symbol, side, entry_price, qty,
            confidence, model_version, features
        )

    def log_trade_exit(self, trade_id, exit_price, exit_reason='UNKNOWN'):
        """Log trade exit."""
        return self.trade_logger.log_exit(trade_id, exit_price, exit_reason)

    def check_and_fine_tune(self):
        """Check performance and fine-tune if needed."""
        is_degraded, metrics, reasons = self.performance_monitor.check_performance()

        if metrics:
            print(f"[MONITOR] Win Rate: {metrics['win_rate']:.1%} | "
                  f"PF: {metrics['profit_factor']:.2f} | "
                  f"Trades: {metrics['total_trades']}")

        if is_degraded:
            print(f"[ALERT] Performance degradation detected!")
            for reason in reasons:
                print(f"  - {reason}")

            # Trigger fine-tuning
            success, message, path = self.auto_retrainer.fine_tune()
            print(f"[FINE-TUNE] {message}")
            return success

        return False

    def check_and_retrain(self):
        """Alias for check_and_fine_tune for backward compatibility."""
        return self.check_and_fine_tune()

    def force_fine_tune(self):
        """Force immediate fine-tuning."""
        print("[MANUAL] Forcing fine-tuning...")
        success, message, path = self.auto_retrainer.fine_tune(force=True)
        print(f"[FINE-TUNE] {message}")
        return success

    def force_retrain(self):
        """Alias for force_fine_tune for backward compatibility."""
        return self.force_fine_tune()

    def get_status(self):
        """Get current system status."""
        status = {
            'total_trades': len(self.trade_logger.trades),
            'open_trades': len(self.trade_logger.get_open_trades()),
            'model_versions': len(self.auto_retrainer.model_history),
            'last_retrain': self.auto_retrainer.last_retrain.isoformat(),
            'is_retraining': self.auto_retrainer.is_retraining,
            'performance': self.performance_monitor.calculate_metrics(
                self.trade_logger.get_recent_trades(PERFORMANCE_WINDOW)
            )
        }
        return status

    def print_status(self):
        """Print formatted status."""
        status = self.get_status()
        print(self.performance_monitor.get_performance_summary())
        print(f"Model versions: {status['model_versions']}")
        print(f"Last retrain: {status['last_retrain']}")
        print(f"Open trades: {status['open_trades']}")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self.check_and_fine_tune()
            except Exception as e:
                print(f"[ERROR] Monitor loop error: {e}")

            # Sleep until next check
            time.sleep(DEGRADATION_CHECK_INTERVAL)

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("[WARN] Monitoring already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("[MONITOR] Background monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        print("[MONITOR] Monitoring stopped")


# ============================================================
# GLOBAL INSTANCE
# ============================================================

# Create global instance for easy access
_manager = None

def get_manager():
    """Get or create the global ContinuousLearningManager instance."""
    global _manager
    if _manager is None:
        _manager = ContinuousLearningManager()
    return _manager


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def log_entry(trade_id, symbol, side, entry_price, qty, confidence, model_version='unknown'):
    """Convenience function to log trade entry."""
    return get_manager().log_trade_entry(
        trade_id, symbol, side, entry_price, qty, confidence, model_version
    )

def log_exit(trade_id, exit_price, exit_reason='UNKNOWN'):
    """Convenience function to log trade exit."""
    return get_manager().log_trade_exit(trade_id, exit_price, exit_reason)

def check_performance():
    """Convenience function to check performance and fine-tune if needed."""
    return get_manager().check_and_fine_tune()

def force_fine_tune():
    """Convenience function to force fine-tuning."""
    return get_manager().force_fine_tune()

def force_retrain():
    """Alias for force_fine_tune for backward compatibility."""
    return force_fine_tune()

def get_status():
    """Convenience function to get status."""
    return get_manager().get_status()

def start_monitoring():
    """Convenience function to start monitoring."""
    return get_manager().start_monitoring()

def stop_monitoring():
    """Convenience function to stop monitoring."""
    return get_manager().stop_monitoring()


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys

    manager = get_manager()

    if len(sys.argv) < 2:
        print("Usage: python continuous_learning.py <command>")
        print("Commands:")
        print("  status    - Show current status")
        print("  check     - Check performance and fine-tune if needed")
        print("  finetune  - Force fine-tuning on recent data")
        print("  monitor   - Start background monitoring")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'status':
        manager.print_status()

    elif command == 'check':
        manager.check_and_fine_tune()

    elif command in ['finetune', 'fine-tune', 'retrain']:
        manager.force_fine_tune()

    elif command == 'monitor':
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        manager.start_monitoring()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            manager.stop_monitoring()
            print("\nMonitoring stopped.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
