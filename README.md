# Multi-Algorithm Stock Predictor

A sophisticated stock price prediction application that uses multiple machine learning algorithms to forecast stock movements. This tool combines the power of LSTM networks, ensemble methods, and technical analysis to provide comprehensive market insights.

## Features

- **Multiple Prediction Models**: Combines LSTM, Random Forest, XGBoost, SVR, KNN, GBM, and ARIMA models for robust predictions
- **Ensemble Approach**: Weighted ensemble of models for more accurate forecasts
- **Technical Analysis**: Calculates key indicators like RSI, MACD, Bollinger Bands
- **Interactive Visualizations**: Candlestick charts with moving averages and prediction visualizations
- **Model Consensus Analysis**: Shows agreement level among different prediction algorithms
- **Risk Assessment**: Evaluates prediction volatility and provides risk level indicators
- **Modern UI**: Clean, responsive interface with intuitive layout

## Installation

Clone the repository:

```bash
git clone https://github.com/Novice-Debugger/BTP.git
cd BTP
```

Create a virtual environment (recommended):

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run stock_predictor.py
```

The application will open in your default web browser at `http://localhost:8501`.

### How to Use

1. Enter a stock symbol (e.g., "AAPL" for Apple)
2. Adjust the historical data range using the slider
3. Click "Generate Predictions" to run the analysis
4. View predictions, technical indicators, and trading signals

## Data Sources

The application uses [Investing.com](https://www.investing.com/) as its primary data source through the `investpy` library. If you encounter connection issues, the application will inform you with appropriate error messages.

## Weight Configurations

The application offers different weight configurations for the ensemble model:

- **Default**: Balanced weights across all models
- **Trend-Focused**: Emphasized weights for trend-following algorithms
- **Statistical**: Higher weights for statistical models
- **Tree-Ensemble**: Prioritizes tree-based models
- **Balanced**: Equal distribution of weights
- **Volatility-Focused**: Optimized for volatile markets

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- Scikit-learn
- XGBoost
- Pandas
- Numpy
- Streamlit
- Plotly
- Investpy

See `requirements.txt` for the complete list of dependencies.

## Troubleshooting

### Data Fetching Issues

If you encounter errors when fetching stock data:

1. Check your internet connection
2. Verify the stock symbol is correct
3. Try using a different stock symbol
4. Wait a few minutes and try again (API rate limits)

### Runtime Errors

- Ensure you have all required dependencies installed
- Check that your Python version is 3.8 or higher
- For TensorFlow errors, ensure your CUDA drivers are up-to-date (for GPU acceleration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses multiple open-source libraries for machine learning and data visualization
- UI design inspired by modern financial dashboards
- Technical analysis calculations based on established financial formulas

## Disclaimer

This application is for educational and informational purposes only. The predictions and analysis should not be considered financial advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
