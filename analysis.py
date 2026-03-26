import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import kmeans
import symnmf

ERROR_OCCURRED = "An Error Has Occurred"
MAX_ITER_DEFAULT = 300  # Default maximum iterations

def main():
    """
    Main function to execute the analysis of K-means and SymNMF algorithms.
    It reads the input data, runs both algorithms for specified values of k, and calculates the
    silhouette scores for each. The results are printed in the required format. 
    If any error occurs during the process, an error message is printed and the program exits.
    """
    np.random.seed(1234)
    if len(sys.argv) < 3:
        print_error(ERROR_OCCURRED)

    k = int(sys.argv[1])
    file_name = sys.argv[2]

    data = read_csv_file(file_name) # Reads the CSV file and returns a list of lists (2D list)
    if not (1 < k < len(data)):
        print_error(ERROR_OCCURRED)

    data_np = np.array(data) # Convert to NumPy array for efficient processing
    
    # labels
    kmeans_labels = get_kmeans_labels(data, k)
    symnmf_labels = get_symnmf_labels(data, k)

    # scores
    try:
        kmeans_score = silhouette_score(data_np, kmeans_labels)
    except ValueError:
        kmeans_score = 0.0 # Assign a default score if silhouette_score fails
    try:
        symnmf_score = silhouette_score(data_np, symnmf_labels)
    except ValueError:
        symnmf_score = 0.0  # Assign a default score if silhouette_score fails

    # results
    print(f"nmf: {symnmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


def get_kmeans_labels(data, k):
    """
    Runs the K-means algorithm and converts the final centroids into a list of cluster labels for each data point.
    This format is required by the silhouette_score function in analysis.py.
    The function calculates the distance from each data point to the centroids and assigns a label based on the closest centroid.
    """
    # Get centroids from existing K-means logic
    dim = len(data[0])
    final_centroids = kmeans.run_kmeans(data, k, MAX_ITER_DEFAULT, dim)
    
    # Convert to NumPy for vectorized operations
    data_np = np.array(data)
    centroids_np = np.array(final_centroids)
    
    # Use broadcasting to compute distances efficiently
    data_expanded = data_np[:, np.newaxis] # Shape: (N, 1, D)
    differences = data_expanded - centroids_np # Shape: (N, K, D)
    
    # Calculate L2 distance and find closest centroid
    distances = np.linalg.norm(differences, axis=2) # Shape: (N, K)
    labels_np = np.argmin(distances, axis=1) # Find min index per row
    
    return labels_np

def get_symnmf_labels(data, k):
    """
    Runs the symnmf algorithm and converts the final H matrix into a list of cluster labels for each data point.
    Similar to the K-means labels, this function calculates the cluster index for each point based 
    on the maximum value in the corresponding row of the H matrix.
    The resulting labels are in the format required by the silhouette_score function in analysis.py.
    """
    # Get H matrix from C extension
    final_h = symnmf.run_goal("symnmf", data, k)
    
    # Convert to NumPy and find cluster index (max value per row)
    # Replaces slow Python loops with efficient C-level argmax
    labels_np = np.argmax(np.array(final_h), axis=1)
    
    return labels_np

def print_error(message):
    """
    Helper function to print an error message and exit the program.
    This centralizes error handling and ensures consistent output when an error occurs.
    """
    print(message)
    sys.exit(1)


def read_csv_file(file_name):
    """
    Reads a CSV file and returns the data as a list of lists (2D list).
    Each inner list represents a row of data.
    On failure prints "An Error Has Occurred" and exits.
    """
    try:
        df = pd.read_csv(file_name, header=None)
        return df.values.tolist()
    except Exception as e:
        print_error(ERROR_OCCURRED)

if __name__ == "__main__":
    main()