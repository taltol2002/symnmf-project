import sys
import os
import numpy as np
import pandas as pd
import symnmfmodule as sm

ERROR_OCCURRED = "An Error Has Occurred"
SYMNMF = "symnmf"
SYM = "sym"
DDG = "ddg"
NORM = "norm"
GOALS = [SYMNMF, SYM, DDG, NORM]

np.random.seed(1234)

def main():
    """
    Main function to execute the SymNMF algorithm based on command line arguments.
    It validates the arguments, reads the input data, and runs the specified goal function.
    The results are printed in the required format. If any error occurs during the process, an error message is printed and the program exits.
    """

    args_data = validate_args(sys.argv)
    if args_data is None:
        print_error(ERROR_OCCURRED)

    k, goal, file_name = args_data

    if not os.path.exists(file_name):
        print_error(ERROR_OCCURRED)

    data = read_csv_file(file_name)
    if data is None or len(data) == 0:
        print_error(ERROR_OCCURRED)
    
    n = len(data)
    if k >= n:
        print_error(ERROR_OCCURRED)
    
    try:
        matrix = run_goal(goal, data, k)
        if matrix is None:
            print_error(ERROR_OCCURRED)
        print_matrix(matrix)
    except Exception:
        print_error(ERROR_OCCURRED)

def initialize_h(W_list, k):
    """
    Initializes the H matrix for the symnmf goal.
    The H matrix is initialized with random values uniformly distributed between 0 and an upper bound.
    The upper bound is calculated based on the mean of the W matrix obtained from the sym function
    in the symnmf module.
    """
    W = np.array(W_list)

    n = W.shape[0]
    m = np.mean(W)

    upper_bound = 2 * np.sqrt(m / k)
    H = np.random.uniform(0, upper_bound, (n,k))
    return H.tolist()

    

def run_goal(goal, data, k):
    """
    Executes the specified goal function on the input data.
    Depending on the goal, it calls the corresponding function from the symnmf module.
    """
    n = len(data)

    if goal == SYM:
        return sm.sym(data)
    if goal == DDG:
        return sm.ddg(data)
    if goal == NORM:
        return sm.norm(data)
    if goal == SYMNMF:
        W = sm.norm(data)  # Compute W using the norm function
        H = initialize_h(W, k)
        return sm.symnmf(H, W, n, k)


def print_matrix(matrix):
    """
    Helper function to print a 2D list (matrix) in the required format.
    Each value is formatted to 4 decimal places and separated by commas.
    """
    for row in matrix:
        print(",".join(format(val, ".4f") for val in row))

def read_csv_file(file_name):
    """
    Reads a CSV file and returns the data as a list of lists (2D list).
    If the file cannot be read, prints an error message and exits the program.
    """
    try:
        df = pd.read_csv(file_name, header=None)
        return df.values.tolist()
    except Exception:
        print_error(ERROR_OCCURRED)

def print_error(msg):
    """
    Helper function to print an error message and exit the program immediately.
    We exit with code 1 to indicate failure.
    """
    print(msg)
    sys.exit(1)


def validate_args(args):
    """
    Validates the command line arguments and returns the values if they are valid.
    If the arguments are invalid, prints an error message and exits the program.
    """
    if len(args) != 4:
        return None
    try:
        k = int(args[1])
        if k <= 0:
            return None
    except ValueError:
        print_error(ERROR_OCCURRED)
    
    goal = args[2]
    if goal not in GOALS:
        print_error(ERROR_OCCURRED)

    file_name = args[3]
    if not file_name.endswith(".txt"):
        return None
    return k, goal, file_name

if __name__ == "__main__":
    main()