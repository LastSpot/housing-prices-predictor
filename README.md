# 🏠 Housing Price Predictor

Ever wondered how much your dream house should cost? Let's predict it! This project uses a fancy neural network to guess house prices (and it's pretty good at it too! 😎).

## ✨ What Makes This Cool?

### 🧹 Smart Data Cleaning
- Handles messy data like a pro
- Fills in missing values using smart neighbors (KNN)
- Turns categories into numbers (one-hot encoding)
- Makes all numbers play nice together (normalization)

### 🧠 Brainy Neural Network
- Built our own brain (MLP) from scratch!
- Uses fancy math (Xavier/Glorot) to start smart
- Learns patterns with sigmoid magic
- Gets better with each try (gradient descent)
- Stays fit with regularization (no overeating!)

### 🎯 Training Like a Pro
- Tries different settings to be the best (grid search)
- Cross-checks itself (k-fold validation)
- Stops when it's good enough (early stopping)
- Keeps itself in check (L2 regularization)

### 🎮 The Knobs We Turn
- Learning speed (alpha)
- Self-control (lambda)
- When to stop (epsilon)
- Network size (layers and neurons)

## 📁 What's in the Box?

```
housing-prices-predictor/
├── model.py                 # Our brain in code
├── housing-classification.ipynb  # The training ground
├── requirements.txt         # The shopping list
├── .gitignore              # The "don't look here" list
└── home-data-for-ml-course/     # The data vault
    ├── train.csv           # Training data
    ├── test.csv            # Test data
    └── data_description.txt # The manual
```

## 🚀 Getting Started

1. Create your virtual playground:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the goodies:
```bash
pip install -r requirements.txt
```

3. Grab the data from [Kaggle](https://www.kaggle.com/competitions/home-data-for-ml-course/data) and drop it in the `home-data-for-ml-course/` folder.

## 🎮 How to Play

1. Fire up `housing-classification.ipynb` in Jupyter Notebook
2. Run all the cells (like pressing play on a movie)
3. Get your predictions in `submission.csv`

## 📊 How Good Is It?

We measure success with Mean Absolute Error (MAE) and:
- Test it 10 different ways (cross-validation)
- Try lots of different settings (grid search)
- Keep it from getting too excited (regularization)

## 🛠️ What You Need

- Python 3.11+ (the newer, the better)
- NumPy (for number crunching)
- Pandas (for data wrangling)
- scikit-learn (for the fancy stuff)
- Jupyter Notebook (for the fun part)

## 📝 License

This project is open source and available under the MIT License. Feel free to use it, share it, and make it even better! 🚀