import sys
import math

ERROR_NUM_CLUSTERS = "Incorrect number of clusters!"
ERROR_MAX_ITER = "Incorrect maximum iteration!"
ERROR_OCCURED = "An Error Has Occurred"
MAX_ITER_DEFAULT = 300  # Default maximum iterations
EPS = 1e-4  # Convergence threshold

def main():
    """
    The main driver of the program. It handles three major tasks:
    1. Parsing: checks command line arguments for K and max_iter, and validates them.
    2. Looping: runs the K-means iteration loop. In every loop, it calls the update functions 
       and checks if the centroids have stopped moving (convergence) or if the max iterations 
       have been reached.
    3. Output: formats the final centroids to 4 decimal places and prints them separated by commas.
    """

    # Parse Command Line Arguments
    # The command is: python3 kmeans.py K [iter] < input_data.txt(other file name)
    
    k, max_iter = parse_args()

    # Read Input Data
    data_points = []
    data_points = read_input()

    # check if document is empty
    if len(data_points) == 0:
        print(ERROR_OCCURED)
        sys.exit(1)

    #Final Validation for K
    N = len(data_points)
    if not (1 < k < N):
        print(ERROR_NUM_CLUSTERS)
        sys.exit(1)

    dim = validate_input(data_points)
    
    # Run the core logic
    final_centroids = run_kmeans(data_points, k, max_iter, dim)
    print_centroids(final_centroids)


def run_kmeans(data_points, k, max_iter, dim):
    """
    This function encapsulates the main K-means algorithm. It initializes the centroids,
    then iteratively updates the clusters and centroids until convergence or reaching the maximum iterations.
    The function returns the final centroids after the algorithm completes.
    """
    arr_centriods = initialize_centroids(data_points, k)
    
    for i in range(max_iter):
        arr_clusters = update_all_clusters(arr_centriods, data_points, k)
        arr_centriods_copy = arr_centriods.copy()
        arr_centriods = update_centers(arr_clusters, data_points, dim)
        
        counter = 0
        for index in range(k):
            distance = compute_distance(arr_centriods[index], arr_centriods_copy[index])
            if distance < EPS:
                counter += 1

        if counter == k:
            break
            
    return arr_centriods


def print_centroids(centroids):
    """
    Helper function to print the centroids in the required format.
    Each centroid's coordinates are formatted to 4 decimal places and separated by commas.
    """
    for centroid in centroids:
        formatted_coords = ['%.4f' % coord for coord in centroid]
        print(','.join(formatted_coords))
        
def parse_args():
    """
    Parses and validates command line arguments. It checks for the correct number of arguments,
    ensures that K is a positive integer, and that max_iter (if provided) is a positive integer within the specified range. 
    If any validation fails, it prints an appropriate error message and exits
    results in a tuple of (K, max_iter) if successful.
    """
    try:
        if len(sys.argv) > 3 or len(sys.argv) < 2:
            print(ERROR_OCCURED)
            sys.exit(1)   

        # Parse K
        try:
            k = int(sys.argv[1])
            if k <= 0:
                print(ERROR_NUM_CLUSTERS)
                sys.exit(1)
        except ValueError:
            # If K is not a NATURAL number
            print(ERROR_NUM_CLUSTERS)
            sys.exit(1)

        # Parse iter (optional)
        max_iter = MAX_ITER_DEFAULT  # Default value
        if len(sys.argv) > 2:
            try:
                max_iter = int(sys.argv[2])
                if max_iter <= 0:
                    print(ERROR_MAX_ITER)
                    sys.exit(1)
            except ValueError:
                print(ERROR_MAX_ITER)
                sys.exit(1)

        # Validation
        # Note: We check 1 < iter < 800 
        if not (1 < max_iter < 800):
            print(ERROR_MAX_ITER)
            sys.exit(1)
        return k, max_iter
    except Exception:
        print(ERROR_OCCURED)
        sys.exit(1)

def update_centers(arr_clusters, data_points, dim):
    """
    This function performs the 'update' step. It calculates the new position for each centroid 
    by finding the average (mean) of all the points currently in its cluster.
    - Normal case: It sums up all the coordinates and divides by the number of points (k).
    - Edge case: If a cluster is empty, the centroid is set to the first input vector (data_points[0]) as required.
      to prevent division by zero errors.
    """

    arr_centroids = []

    for clusterX in arr_clusters:
        k = len(clusterX)
        sum_point = [0.0 for i in range(dim)]
        for point in clusterX:
            for i in range(dim):
                sum_point[i]+=point[i]
        centroidX = [0.0 for i in range(dim)]        

        for i in range(len(sum_point)):
            if k>0:
                centroidX[i] = sum_point[i]/k
            else:
                centroidX = data_points[0]
                break      
        arr_centroids.append(centroidX)
    return arr_centroids


def update_all_clusters(arr_centriods,data_points,k):
    """
    This function is responsible for the 'assignment' step. It creates k empty groups (clusters). 
    Then, it looks at every single data point, calculates its distance to all available centroids, 
    finds the closest one, and adds the point to that centroid's group.
    """
    
    arr_clusters = []
    for i in range(k):
        arr_clusters.append([])

    for vectorX in data_points:
        min_distance = compute_distance(arr_centriods[0], vectorX)
        number_of_min_centr = 0
        number_of_curr_centr = -1
        for centroidX in arr_centriods:
            number_of_curr_centr+=1
            distance = compute_distance(vectorX, centroidX)
            if distance < min_distance:
                min_distance = distance
                number_of_min_centr = number_of_curr_centr
        arr_clusters[number_of_min_centr].append(vectorX)
        
    return arr_clusters


def initialize_centroids(data_points, k):   
    """
    Sets up the starting state of the algorithm. As per the requirements, this function simply 
    grabs the first k points from the input data and uses them as the initial centroids.
    """

    return data_points[:k]

def compute_distance(point1, point2): 
    """
    Calculates the Euclidean distance between two vectors (points). It iterates through the 
    coordinates of both points, sums up the squared differences, and returns the square root 
    of that sum.
    """

    sum_dist = 0
    for i in range(len(point1)):
        sum_dist += (point1[i]-point2[i])**2
    
    distance = math.sqrt(sum_dist)
    return distance


def read_input():
    """
    Reads the raw text data from the standard input. It goes through the input line by line, 
    splits the text by commas, converts the numbers to floats, and stores them as vectors in a list. 
    If anything goes wrong during reading, it prints an error and exits.
    """

    data_points = []
    try:
        for line in sys.stdin:
            if not line.strip():  # Skip empty lines
                continue
            vector = [float(x) for x in line.split(',')]
            data_points.append(vector)
    except Exception:
        print(ERROR_OCCURED)
        sys.exit(1)
    return data_points


def validate_input(data_points):
    """
    Validates that:
    1) There is at least 1 data point.
    2) All data points have the same dimension.
    Returns the dimension (d) if valid.
    On failure prints "An Error Has Occurred" and exits.
    """
    if len(data_points) == 0:
        print(ERROR_OCCURED)
        sys.exit(1)

    dim = len(data_points[0])
    if dim == 0:
        print(ERROR_OCCURED)
        sys.exit(1)

    for vec in data_points:
        if len(vec) != dim:
            print(ERROR_OCCURED)
            sys.exit(1)

    return dim

if __name__ == "__main__":
    main()
    exit(0)