# 🏠 Housing Price Predictor

Ever wondered how much your dream house should cost? Let's predict it! This project uses a fancy neural network I built to guess house prices (and it's pretty good at it too! 😎).

## ✨ What Makes This Cool?

### 🧹 Smart Data Cleaning
- Handles messy data like a pro
- Fills in missing values using smart neighbors (KNN)
- Turns categories into numbers (one-hot encoding)
- Makes all numbers play nice together (normalization)

### 🧠 Brainy Neural Network
- Built my own brain (MLP) from scratch!
- Uses fancy math (Xavier/Glorot) to start smart
- Learns patterns with sigmoid magic
- Gets better with each try (gradient descent)
- Stays fit with regularization (no overeating!)
- Auto-loads best parameters on startup
- Saves & loads your training results 💾

### 🎯 Training Like a Pro
- Tries different settings to be the best (grid search)
- Cross-checks itself (k-fold validation)
- Stops when it's good enough (early stopping)
- Keeps itself in check (L2 regularization)
- Remembers what worked best (parameter persistence)
- Smart enough to skip bad parameter combinations (saves hours!)

### 🎮 The Knobs I Turn
- Learning speed (alpha)
- Self-control (lambda)
- When to stop (epsilon)
- Network size (layers and neurons)

### 🔍 Model Transparency
- Detailed model information with `get_info()`
- Parameter summaries with `summarize_parameters()`
- Easy model creation from saved parameters
- Parameter file management and inspection

## 📁 What's in the Box?

```
housing-prices-predictor/
├── model.py                 # My brain in code
├── housing-classification.ipynb  # The training ground
├── requirements.txt         # The shopping list
├── .gitignore              # The "don't look here" list
├── best_params.pkl         # My saved model parameters
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

## 💾 Parameter Management

No need to retrain every time! My model now supports:

```python
# Create a new model (automatically loads parameters if they exist)
model = MLPNeuralNetwork(input_size=288, output_size=1)

# Create a model directly from saved parameters
model = MLPNeuralNetwork.create_from_parameters("best_params.pkl")

# Check available parameter files
MLPNeuralNetwork.check_saved_parameters()

# See what's inside your model
model.get_info()
model.summarize_parameters()
```

## ⚡ Smart Grid Search

Speed up your grid search with early stopping criteria:

```python
# Define your parameter grid
params = {
    'alphas': [0.001, 0.01, 0.1],
    'lambdas': [0, 0.01, 0.1],
    'epsilons': [1e-8, 1e-6],
    'hidden_sizes': [1, 2, 3],
    'neurons_per_layer': [10, 20, 30],
    # Early stopping parameters
    'early_stopping_threshold': 20000,  # Skip combinations with MAE > 20000
    'early_stopping_folds': 2           # After testing 2 folds
}

# Run grid search with early stopping
model.grid_search(X_train, y_train, params, kf)
```

This lets you skip poor parameter combinations after just a few folds, saving hours of computation time!

## 📊 How Good Is It?

I measure success with Mean Absolute Error (MAE) and:
- Test it 10 different ways (cross-validation)
- Try lots of different settings (grid search)
- Keep it from getting too excited (regularization)
- Save the best parameters for future use

## 🛠️ What You Need

- Python 3.11+ (the newer, the better)
- NumPy (for number crunching)
- Pandas (for data wrangling)
- scikit-learn (for the fancy stuff)
- Jupyter Notebook (for the fun part)

## 📝 License

This project is open source and available under the MIT License. Feel free to use it, share it, and make it even better! 🚀