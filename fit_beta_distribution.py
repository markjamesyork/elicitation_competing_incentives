import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def fit_beta_distribution(data_points):
    # Fit a beta distribution to the data points
    # The `beta.fit` method returns alpha, beta, loc (usually 0 for a beta distribution), and scale (usually 1)
    # We use floc=0 and fscale=1 to fix the location and scale to standard beta distribution bounds
    alpha, beta, loc, scale = stats.beta.fit(data_points, floc=0, fscale=1)

    # Print the estimated parameters
    print(f"Estimated alpha: {alpha}")
    print(f"Estimated beta: {beta}")

    # Generate values from the fitted distribution
    fitted_values = stats.beta.pdf(np.linspace(0, 1, num=100), alpha, beta)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.hist(data_points, bins=30, density=True, alpha=0.5, label='Average Reports per Borrower')
    plt.plot(np.linspace(0, 1, num=100), fitted_values, 'r-', lw=3, label='Fitted Beta Distribution')
    plt.title('Average Reports per Borrower and Fitted Beta Distribution')
    plt.legend()
    plt.show()

    return alpha, beta

# Execution
#data_points = np.random.beta(2, 5, size=1000)  # Generate some synthetic data for demonstration
df = pd.read_csv('csv_files/average_reports.csv')
data_points = df.to_numpy()
print(data_points)
alpha, beta = fit_beta_distribution(data_points)
print('alpha; ', alpha)
print('beta: ', beta)