import numpy as np
import pandas as pd


def generate_random_covariance_matrix_and_mean(size, dataNum):
    """
    Generates a random covariance matrix of the given size.
    """

    random_data = 1 * np.random.rand(size, dataNum)
    covariance_matrix = np.cov(random_data, rowvar=True)
    mean_vector = np.mean(random_data, axis=1)

    return covariance_matrix, mean_vector


# Example usage
size = 10
cov, mean = generate_random_covariance_matrix_and_mean(size, 100)

# Create indices as "prod0", "prod1", ..., "prod<size-1>"
indices = ["prod" + str(i) for i in range(size)]

# Store covariance matrix as CSV
cov_df = pd.DataFrame(cov, index=indices, columns=indices)
cov_df.to_csv("generated/covariance_matrix.csv")

# Store mean vector as CSV
mean_df = pd.DataFrame(mean, index=indices, columns=["mean"])
mean_df.to_csv("generated/mean_vector.csv")