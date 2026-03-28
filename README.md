# High-Frequency Volatility Transformer

Stop forecasting market volatility with antiquated math from the 1990s. This repository implements a robust deep-learning pipeline that ingests intraday SPY and VIX data to predict 7-hour rolling realized volatility. 

It ditches standard sequential models in favor of a PyTorch-based Transformer paired with a custom `Time2Vec` embedding layer. It also features an asymmetric loss function—because underestimating market panic is a great way to blow up your portfolio, and we prefer keeping our capital.

## Core Components

* **Microstructure Data Pipeline (`data_pipeline.py`):** Pulls 720 days of 1-hour bar data for SPY and ^VIX via `yfinance`. Calculates hourly log returns and a precise 7-row rolling target (exactly one full trading session).
* **The Alien Brain (`transformer_model.py`):** A custom PyTorch Multi-Head Attention Transformer. It utilizes a `Time2Vec` layer to learn fluid time frequencies—blending linear trends with periodic cyclical sine waves—and injects this learned clock directly into the financial data.
* **The Thunderdome (`train.py`):** The training loop. Uses `Optuna` for hyperparameter tuning across multiple architecture variants. We batch the data into DataLoaders (the "Drip Feed") because exploding your RAM is generally considered bad practice.
* **Asymmetric Volatility Loss:** Custom loss module that applies a 3x penalty to under-predictions. 
* **Walk-Forward Validation:** Executes rolling walk-forward validation across multiple eras to simulate real-world, out-of-sample deployment.

## Traditional Models vs. Our Transformer

| Feature | Standard Quant Approach | This Repository |
| :--- | :--- | :--- |
| **Temporal Encoding** | Fixed Sine/Cosine Positional Encoding | **Time2Vec** (Learnable linear/periodic embeddings) |
| **Architecture** | GARCH or LSTM | **Multi-Head Attention Transformer** |
| **Loss Function** | Mean Squared Error (MSE) | **Asymmetric Loss** (3x penalty for under-predicting risk) |
| **Hyperparameter Tuning** | Grid Search / Guessing | **Optuna TPE Sampler** |

## Installation & Execution

You will need PyTorch, Optuna, and a machine that doesn't belong in a museum. 

1.  **Clone and Install Dependencies:** Ensure you have `yfinance`, `pandas`, `numpy`, `scikit-learn`, `torch`, and `optuna`.
2.  **Generate the Parquet File:**
    ```bash
    python data_pipeline.py
    ```
3.  **Train the Matrix:**
    ```bash
    python train.py
    ```

## Output 

The pipeline outputs an optimized set of hyperparameters, completes the walk-forward validation across defined eras, and yields an array of descaled volatility predictions ready for backtesting or active trading logic.