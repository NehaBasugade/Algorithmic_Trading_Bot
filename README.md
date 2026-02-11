# Algorithmic Trading Bot 
Comparative Study of Rule-Based, Machine Learning, and Reinforcement Learning Strategies

This project is my **Final Year Research Project (RIT, 2023)** focused on building and comparing three different approaches to algorithmic trading using historical stock market data.

The goal was not just to maximize returns, but to understand how **traditional technical analysis**, **machine learning**, and **reinforcement learning** behave under the same market conditions.

---

## Project Overview
I implemented and evaluated **three trading strategies** on historical stock data (SBIN.NS):

1. **Manual Technical Analysis**
2. **Machine Learning with Technical Indicators**
3. **Reinforcement Learning with Technical Indicators**

Each approach uses the same underlying price data but differs in how trading decisions are made.

---

## 1. Manual Technical Analysis Strategy
**Notebook:** `CAPSTONE MANUAL METHOD.ipynb`

This approach uses classic trading indicators and rule-based logic.

### Indicators used
- MACD (26, 12, 9)
- Stochastic Oscillator (%K, %D)

### How it works
- Buy and sell signals are generated based on indicator crossovers and threshold rules
- Positions are tracked manually
- Strategy returns are calculated and compared against market returns
- Profit and percentage return are computed for a fixed investment amount

### What this showed
- Rule-based strategies are simple and interpretable
- Performance is highly sensitive to indicator thresholds
- Works well in trending markets, struggles in sideways conditions

---

## 2. Machine Learning with Technical Indicators
**Notebook:** `CAPSTONE ML METHOD.ipynb`

This approach frames trading as a **classification problem**.

### Features engineered
- Price-based features (Open–Close, High–Low)
- Moving averages (3-day, 10-day, 30-day)
- Volatility (standard deviation)
- Technical indicators:
  - RSI
  - Williams %R

### Model used
- Feedforward Neural Network (Keras / TensorFlow)

### How it works
- Predicts whether the price will rise the next day
- Converts predictions into buy/sell decisions
- Compares cumulative strategy returns vs market returns

### What this showed
- Feature engineering has a huge impact on results
- Neural networks can capture non-linear patterns
- Overfitting is a real risk in financial time series
- ML models need careful validation and scaling

---

## 3. Reinforcement Learning with Technical Indicators
**Notebook:** `CAPSTONE RL METHOD.ipynb`

This approach treats trading as a **sequential decision-making problem**.

### Environment
- `gym-anytrading` (customized stock environment)

### State features
- Price data
- SMA
- RSI
- OBV
- Volume

### Algorithm used
- **Advantage Actor-Critic (A2C)** from Stable-Baselines3

### How it works
- The agent learns when to buy, sell, or hold based on rewards
- Rewards are tied to trading performance
- The agent is trained over multiple timesteps and then evaluated visually

### What this showed
- Reinforcement learning can adapt to changing market behavior
- Training is computationally expensive
- Reward design strongly influences strategy quality
- RL strategies are harder to interpret but more flexible

---

## Data
- Source: **Yahoo Finance**
- Stock used: `SBIN.NS`
- Period: **2017 – 2023**
- Prices adjusted for splits/dividends

---

## Tech Stack
**Core**
- Python
- NumPy, Pandas
- Matplotlib

**Finance & Data**
- yfinance
- TA-Lib
- Finta

**Machine Learning**
- scikit-learn
- TensorFlow / Keras

**Reinforcement Learning**
- OpenAI Gym
- gym-anytrading
- Stable-Baselines3 (A2C)

---

## What I Learned from This Project
- How different trading paradigms behave on the same data
- Strengths and weaknesses of rule-based vs learning-based systems
- Importance of feature engineering in financial ML
- How technical indicators translate into model inputs
- Designing rewards for reinforcement learning agents
- Why backtesting alone is not enough for real trading systems
- How noise and non-stationarity affect financial models

---

## Results & Insights
- No single strategy consistently outperformed in all market conditions
- Manual strategies are interpretable but rigid
- ML strategies offer better adaptability but risk overfitting
- RL strategies are powerful but complex and resource-intensive
- Combining domain knowledge with learning methods gives the best results

---

## Future Improvements
- Add transaction costs and slippage
- Perform walk-forward validation
- Try LSTM / Transformer-based models
- Improve RL reward functions
- Extend to portfolio-level trading
- Deploy strategies in a paper-trading environment

---

## Note
This project is **academic and research-focused**.  
It is not intended for live trading without further validation and risk controls.
