import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna
from transformer_model import VolatilityTransformer

torch.manual_seed(42)
np.random.seed(42)

# The Wall Street Judge survives
class AsymmetricVolatilityLoss(nn.Module):
    def __init__(self, penalty_factor=3.0):
        super().__init__()
        self.penalty_factor = penalty_factor

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        squared_error = error ** 2
        multiplier = torch.where(error < 0, self.penalty_factor, 1.0)
        return torch.mean(squared_error * multiplier)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, 1])
    return np.array(xs), np.array(ys)

def run_optimized_pipeline(file_path="SPY_TRANSFORMER_clean.parquet", n_trials=10, n_splits=4):
    df = pd.read_parquet(file_path)
    seq_length = 21
    
    test_start_idx = int(len(df) * 0.8)
    optuna_train_df = df.iloc[:test_start_idx]
    test_df = df.iloc[test_start_idx:]
    
    print("1. Prepping Tensor Data for the Attention Matrix...")
    scaler = StandardScaler()
    features = ['log_return', 'realized_vol', 'vix_close']
    train_scaled = scaler.fit_transform(optuna_train_df[features])
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    print(f"2. The Thunderdome (Running {n_trials} Transformer Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    def objective(trial):
        # The math constraint: d_model must be cleanly divisible by nhead
        d_model = trial.suggest_categorical('d_model', [32, 64])
        nhead = trial.suggest_categorical('nhead', [2, 4])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.2, 0.4)
        lr = trial.suggest_float('lr', 0.0005, 0.005)
        
        model = VolatilityTransformer(input_size=3, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        # THE FIX: weight_decay to stop the Transformer from overfitting
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) 
        criterion = AsymmetricVolatilityLoss(penalty_factor=3.0)
        
        for _ in range(30): 
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
        return loss.item()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    print(f"--> Multi-Head Architecture Locked: {best}")

    print(f"3. Executing Rolling Walk-Forward Validation ({n_splits} Eras)...")
    step_size = len(test_df) // n_splits
    all_descaled_preds = []
    
    train_window_size = test_start_idx
    
    for step in range(n_splits):
        print(f"   -> Retraining the Attention Matrix: Era {step + 1}/{n_splits}...")
        current_train_end = test_start_idx + (step * step_size)
        current_test_end = current_train_end + step_size if step < n_splits - 1 else len(df)
        
        window_start = current_train_end - train_window_size
        fold_train_df = df.iloc[window_start:current_train_end]
        fold_test_df = df.iloc[current_train_end - seq_length:current_test_end]
        
        fold_scaler = StandardScaler()
        fold_train_scaled = fold_scaler.fit_transform(fold_train_df[features])
        fold_test_scaled = fold_scaler.transform(fold_test_df[features])
        
        f_X_train, f_y_train = create_sequences(fold_train_scaled, seq_length)
        f_X_test, _ = create_sequences(fold_test_scaled, seq_length)
        
        f_X_train_t = torch.tensor(f_X_train, dtype=torch.float32)
        f_y_train_t = torch.tensor(f_y_train, dtype=torch.float32).unsqueeze(1)
        f_X_test_t = torch.tensor(f_X_test, dtype=torch.float32)
        
        torch.manual_seed(42)
        fold_model = VolatilityTransformer(
            input_size=3, 
            d_model=best['d_model'], 
            nhead=best['nhead'], 
            num_layers=best['num_layers'], 
            dropout=best['dropout']
        )
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=best['lr'], weight_decay=1e-5)
        criterion = AsymmetricVolatilityLoss(penalty_factor=3.0)
        
        for epoch in range(40):
            fold_model.train()
            fold_optimizer.zero_grad()
            loss = criterion(fold_model(f_X_train_t), f_y_train_t)
            loss.backward()
            fold_optimizer.step()
            
        fold_model.eval()
        with torch.no_grad():
            preds_scaled = fold_model(f_X_test_t).numpy()
            
        dummy = np.zeros((len(preds_scaled), 3))
        dummy[:, 1] = preds_scaled[:, 0]
        preds_descaled = fold_scaler.inverse_transform(dummy)[:, 1]
        all_descaled_preds.extend(preds_descaled)

    final_wf_predictions = np.array(all_descaled_preds)
    # Notice we dropped the GARCH return entirely. We are playing strictly with deep learning today.
    return test_df, final_wf_predictions, best