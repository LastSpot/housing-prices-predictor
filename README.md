# 🏠 Housing Price Predictor

Ever wondered how much your dream house should cost? Let's predict it! This project uses a custom neural network to predict house prices with high accuracy.

## ✨ What Makes This Cool?

### 🧹 Smart Data Cleaning
- Handles messy data like a pro
- Fills in missing values using smart neighbors (KNN)
- Turns categories into numbers (one-hot encoding)
- Makes all numbers play nice together (normalization)
- Consistent encoding across training and test data

### 🧠 Brainy Neural Network
- Built a Multi-Layer Perceptron (MLP) from scratch!
- Uses He initialization for better performance
- Learns patterns with sigmoid activation
- Gets better with each try (gradient descent)
- Stays fit with regularization (no overeating!)
- Auto-loads best parameters on startup
- Saves & loads your training results 💾
- Robust error handling to prevent NaN issues

### 📈 Smart Loss Functions
- **RMSE**: Standard Root Mean Squared Error for general prediction
- **Weighted RMSE**: Gives more importance to expensive houses (reduces % error)
- **Huber Loss**: Combines MSE and MAE to be less sensitive to outliers
- **Log-Cosh Loss**: Smooth approximation of Huber loss with better gradients
- **MAE**: Mean Absolute Error option for straightforward average error

### 🎯 Training Like a Pro
- Tries different settings to be the best (grid search)
- Cross-checks itself (k-fold validation)
- Stops when it's good enough (early stopping)
- Keeps itself in check (L2 regularization)
- Remembers what worked best (parameter persistence)
- Smart enough to skip bad parameter combinations (saves hours!)
- Evaluates using customizable loss functions for better accuracy

### 🎮 The Knobs I Turn
- Learning speed (alpha)
- Self-control (lambda)
- When to stop (epsilon)
- Network size (layers and neurons)
- Loss function type (RMSE, weighted RMSE, Huber, Log-Cosh, MAE)

### 🔍 Model Transparency
- Detailed model information with `get_info()`
- Parameter summaries with `summarize_parameters()`
- Easy model creation from saved parameters
- Parameter file management and inspection

## 📁 What's in the Box?

```
housing-prices-predictor/
├── model.py                     # The neural network implementation
├── housing-prediction-script.py # The main processing script
├── housing-classification.ipynb # Interactive Jupyter notebook
├── requirements.txt             # Dependencies
├── .gitignore                   # Ignored files
├── best_params.pkl              # Saved model parameters
└── home-data-for-ml-course/     # Data directory
    ├── train.csv                # Training data
    ├── test.csv                 # Test data
    └── data_description.txt     # Feature descriptions
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

1. Fire up the script:
```bash
python housing-prediction-script.py
```

Or use the notebook:
```bash
jupyter notebook housing-classification.ipynb
```

2. Get your predictions in `submission.csv`

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
    'alphas': [0.6, 0.7, 0.8, 0.9],
    'lambdas': [0, 0.1, 0.2, 0.3],
    'epsilons': [math.pow(math.e, -7), math.pow(math.e, -6), math.pow(math.e, -5)],
    'hidden_sizes': [5, 10],
    'neurons_per_layer': [10, 20],
    # Early stopping parameters
    'early_stopping_threshold': 80000,  # Skip combinations with MAE > 20000
    'early_stopping_folds': 5           # After testing 2 folds
}

# Run grid search with early stopping
model.grid_search(X_train, y_train, params, k_fold_index)
```

This lets you skip poor parameter combinations after just a few folds, saving hours of computation time!

## 🎯 Custom Loss Functions

The model now supports multiple loss functions to fine-tune prediction accuracy:

```python
# Grid search with weighted RMSE (weights errors on expensive houses more heavily)
model.grid_search(X_train, y_train, params, k_fold_index, loss_type='weighted_rmse')

# Train with Huber loss (combines MSE and MAE, less sensitive to outliers)
model.fit(X_train, y_train, loss_type='huber')

# Test with log-cosh loss (smooth approximation of Huber)
score = model.test(X_test, y_test, loss_type='log_cosh')
```

Different loss functions are optimal for different situations:

- **weighted_rmse**: Best when accuracy on expensive houses matters more
- **huber**: Best when your data has outliers (unusually expensive houses)
- **log_cosh**: Smoothest training with differentiable gradients
- **rmse**: Standard metric used in most competitions
- **mae**: When you want to minimize average absolute dollar error

Parameters are saved with the loss function name included (e.g., `best_params_huber.pkl`).

## 📊 How Good Is It?

I measure success with multiple loss metrics, which:
- Penalize errors differently based on the problem needs
- Give a better sense of prediction quality for housing prices
- Are in the same unit as the target variable (dollars)

The model:
- Tests across multiple data splits (k-fold cross-validation)
- Tries numerous parameter combinations (grid search)
- Prevents overfitting through regularization
- Saves the best parameters for future use
- Uses specific loss functions optimized for real estate pricing

## 🛠️ What You Need

- Python 3.11+ (the newer, the better)
- NumPy (for number crunching)
- Pandas (for data wrangling)
- scikit-learn (for data processing utilities)
- Jupyter Notebook (optional, for interactive exploration)

## 📝 License

This project is open source and available under the MIT License. Feel free to use it, share it, and make it even better! 🚀