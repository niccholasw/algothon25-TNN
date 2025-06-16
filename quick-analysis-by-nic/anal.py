# DEEPSEEK CODE - GENERATED FOR QUICK SIMPLE ANALYSIS BY NIC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(file_path):
    """Load data from raw text file and process into DataFrame"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert each line to a list of floats
    data = []
    for line in lines:
        row = list(map(float, line.strip().split()))
        data.append(row)
    
    # Transpose so columns are assets and rows are days
    data = np.array(data).T
    df = pd.DataFrame(data)
    
    # Name columns and index
    df.columns = [f'Day {i+1}' for i in range(df.shape[1])]
    df.index = [f'Asset {i+1}' for i in range(df.shape[0])]
    
    return df

def analyze_data(df):
    """Perform all required analyses on the data"""
    results = {}
    
    # 1. Price trajectories per asset
    plt.figure(figsize=(12, 6))
    for asset in df.index:
        plt.plot(df.columns, df.loc[asset], label=asset)
    plt.title('Price Trajectories of All Assets')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    results['price_trajectories'] = plt.gcf()
    plt.show()
    
    # 2. Volatility and average daily return per asset
    returns = df.diff(axis=1).iloc[:, 1:] / df.iloc[:, :-1].values
    volatility = returns.std(axis=1) * np.sqrt(252)  # Annualized volatility
    avg_return = returns.mean(axis=1) * 252  # Annualized return
    
    results['volatility'] = volatility
    results['avg_return'] = avg_return
    
    plt.figure(figsize=(12, 6))
    plt.bar(volatility.index, volatility)
    plt.title('Volatility (Annualized) per Asset')
    plt.xlabel('Asset')
    plt.ylabel('Volatility')
    plt.xticks(rotation=90)
    plt.tight_layout()
    results['volatility_plot'] = plt.gcf()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.bar(avg_return.index, avg_return)
    plt.title('Average Daily Return (Annualized) per Asset')
    plt.xlabel('Asset')
    plt.ylabel('Average Return')
    plt.xticks(rotation=90)
    plt.tight_layout()
    results['return_plot'] = plt.gcf()
    plt.show()
    
    # 3. Correlations between assets
    correlation_matrix = df.T.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Between Assets')
    plt.tight_layout()
    results['correlation_plot'] = plt.gcf()
    plt.show()
    
    # 4. Flagging stable vs. volatile instruments
    # Define stable as volatility in bottom quartile, volatile in top quartile
    vol_q1 = volatility.quantile(0.25)
    vol_q3 = volatility.quantile(0.75)
    
    stable_assets = volatility[volatility <= vol_q1].index.tolist()
    volatile_assets = volatility[volatility >= vol_q3].index.tolist()
    
    results['stable_assets'] = stable_assets
    results['volatile_assets'] = volatile_assets
    
    # Create scatter plot of return vs volatility
    plt.figure(figsize=(10, 6))
    plt.scatter(volatility, avg_return)
    
    # Annotate stable and volatile assets
    for asset in stable_assets:
        plt.annotate(asset, (volatility[asset], avg_return[asset]), color='green')
    for asset in volatile_assets:
        plt.annotate(asset, (volatility[asset], avg_return[asset]), color='red')
    
    plt.title('Risk-Return Profile of Assets')
    plt.xlabel('Volatility')
    plt.ylabel('Average Return')
    plt.axvline(vol_q1, color='gray', linestyle='--')
    plt.axvline(vol_q3, color='gray', linestyle='--')
    plt.tight_layout()
    results['risk_return_plot'] = plt.gcf()
    plt.show()
    
    return results

def save_results(results, output_file='analysis_results.txt'):
    """Save analysis results to a text file"""
    with open(output_file, 'w') as f:
        f.write("Financial Instrument Analysis Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Stable Assets (Low Volatility):\n")
        f.write(", ".join(results['stable_assets']) + "\n\n")
        
        f.write("Volatile Assets (High Volatility):\n")
        f.write(", ".join(results['volatile_assets']) + "\n\n")
        
        f.write("Volatility Metrics:\n")
        f.write(results['volatility'].to_string() + "\n\n")
        
        f.write("Average Return Metrics:\n")
        f.write(results['avg_return'].to_string() + "\n\n")
        
        f.write("Correlation Analysis:\n")
        f.write("See correlation matrix plot for details\n")

def main():
    # Assuming data is in 'instrument_data.txt'
    file_path = './prices.txt'
   
    # Load and process datai
    df = load_and_process_data(file_path)
    
    # Analyze data
    results = analyze_data(df)
    
    # Save results
    save_results(results)
    
    print("Analysis complete. Results saved to 'analysis_results.txt'")
    print(f"Stable assets: {results['stable_assets']}")
    print(f"Volatile assets: {results['volatile_assets']}")

if __name__ == "__main__":
    main()