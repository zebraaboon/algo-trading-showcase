# Algorithmic Trading Toolkit - Code Samples

> Building blocks for cryptocurrency trading systems.
> These are sanitised examples from my own trading research platform (98,000+ lines, 6 exchanges, 60 real-time feeds).

## About Me

I build **multi-exchange algorithmic trading systems** in Python. My own system:

- Processes **60 real-time L2 order book feeds** across 6 exchanges (Coinbase, Binance, Kraken, Bybit, OKX, Bitstamp)
- Trains **XGBoost ML models** on 18M+ candles with walk-forward validation
- Runs **24/7 on dedicated hardware** with circuit breakers and auto-recovery
- Supports **5 currencies** (GBP, USD, EUR, USDT, USDC) with real-time FX conversion

## Code Samples

| File | Description | Key Skills |
|------|-------------|------------|
| [`websocket_orderbook.py`](websocket_orderbook.py) | Real-time L2 order book processor with reconnection logic | asyncio, WebSocket, data pipelines |
| [`feature_engineering.py`](feature_engineering.py) | Microstructure feature extraction from order book data | NumPy, pandas, quantitative finance |
| [`walk_forward_cv.py`](walk_forward_cv.py) | Walk-forward cross-validation for time-series ML | XGBoost, scikit-learn, backtesting |
| [`risk_manager.py`](risk_manager.py) | Position sizing with Kelly criterion and circuit breakers | Risk management, async Python |
| [`exchange_interface.py`](exchange_interface.py) | Abstract exchange adapter with rate limiting | API design, async patterns, OOP |

## Tech Stack

**Core**: Python 3.11+, asyncio, aiohttp, NumPy, pandas

**ML/AI**: XGBoost, scikit-learn, SHAP, Temporal Fusion Transformers (PyTorch)

**Infrastructure**: SQLite, WebSocket, REST APIs, systemd, Tailscale

**Exchanges**: Coinbase Advanced Trade API, Binance, Kraken, Bybit, OKX, Bitstamp

## Architecture Highlights

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  60 WebSocket │────▶│   Feature    │────▶│  ML Models   │
│    Feeds      │     │  Pipeline    │     │  (XGBoost)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   Exchange   │◀────│  Execution   │◀────│   Strategy   │
│   Adapters   │     │   Engine     │     │   Manager    │
└──────────────┘     └──────────────┘     └──────────────┘
        │                                         │
        ▼                                         ▼
┌──────────────┐                          ┌──────────────┐
│    Risk      │                          │   Portfolio   │
│   Manager    │                          │   Optimizer   │
└──────────────┘                          └──────────────┘
```

## Contact

**Upwork**: [upwork.com/freelancers/~01a24749cb521283a8](https://www.upwork.com/freelancers/~01a24749cb521283a8)

Available for:
- Custom trading bot development
- Exchange API integration
- ML model training and validation
- Backtesting frameworks
- Real-time data pipelines

## License

These code samples are provided for demonstration purposes. Not financial advice.
