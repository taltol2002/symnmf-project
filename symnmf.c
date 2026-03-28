#include "symnmf.h"

/* This block ensures the string exists in the .c file for static testers */
#ifndef ERROR_MSG
#define ERROR_MSG "An Error Has Occurred"
#endif

/* 
 * Allocates a 2D matrix of doubles with given dimensions.
 * Exits with an error message if memory allocation fails.
 * Caller is responsible for freeing the allocated memory.
 */
double** allocate_matrix(int rows, int cols)
{
    int i;
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) {
        printf("%s\n", ERROR_MSG);
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix[i]) {
            free_matrix(matrix, i); /* Free previously allocated rows */
            printf("%s\n", ERROR_MSG);
            return NULL;
        }
    }
    return matrix;
}

/*
 * Frees a 2D matrix of doubles.
 * Caller is responsible for ensuring the matrix is properly allocated.
 */
void free_matrix(double** mat, int rows){
    int i;
    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

/*
 * Copies values from source matrix to destination matrix.
 * Both matrices must have the same dimensions (n x k).
 */
void copy_matrix_values(double** dest, double** src, int n, int k) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

/*
 * Frees a list of vectors.
 * Caller is responsible for ensuring the list is properly allocated.
 */
void free_vector_list(vector* head) {
    vector* temp;
    while (head != NULL) {
        temp = head;
        head = head->next;
        if (temp->data != NULL) {
            free(temp->data);
        }
        free(temp);
    }
}

/*
 * Resizes a buffer for reading data from a file.
 * The buffer is doubled in size when the capacity is exceeded.
 * Exits with an error message if memory allocation fails.
 */
double* resize_buffer(double* buffer, int* capacity) {
    double* new_buf;

    *capacity *= 2;
    new_buf = realloc(buffer, *capacity * sizeof(double));
    if (!new_buf) {
        printf("%s\n", ERROR_MSG);
        return NULL;
    }
    return new_buf;
}

/* 
 * Creates a new vector node with the given data.
 * Exits with an error message if memory allocation fails.
 */
vector* create_node(double* temp_row, int d) {
    int i;
    vector* newNode = malloc(sizeof(vector));
    if (!newNode) {
        printf("%s\n", ERROR_MSG);
        return NULL;
    }
    
    newNode->data = malloc(d * sizeof(double));
    if (!newNode->data) {
        printf("%s\n", ERROR_MSG);
        free(newNode);
        return NULL;
    }
    
    for (i = 0; i < d; i++) {
        newNode->data[i] = temp_row[i];
    }
    newNode->next = NULL;
    return newNode;
}

/*
 * Helper function to append a new vector node to the linked list.
 * Returns the new tail node, or NULL on memory allocation failure.
 */
vector* append_row(vector **head, vector *curr, double *buf, int d) {
    vector *new_node = create_node(buf, d);
    if (!new_node) return NULL;
    
    if (!*head) {
        *head = new_node; /* First node in the list */
    } else {
        curr->next = new_node; /* Append to the end */
    }
    return new_node;
}

/*
 * Loads data from a file into a linked list of vectors.
 * Each line in the file represents a vector, with values separated by spaces.
 * The function updates n and d to reflect the number of vectors and their dimensionality.
 * Exits with an error message if memory allocation fails.
 */
vector* load_data(FILE* f, int* n, int* d) {
    double val; char c; int cap = 1024, col = 0, res;
    vector *head = NULL, *curr = NULL;
    double *buf = malloc(cap * sizeof(double));
    if (!buf) {
        printf("%s\n", ERROR_MSG); 
        return NULL; 
    }
    *n = 0; *d = 0;
    while ((res = fscanf(f, "%lf%c", &val, &c)) >= 1) {
        if (col >= cap) buf = resize_buffer(buf, &cap);
        if(!buf) {
            free_vector_list(head);
            return NULL;
        }
        buf[col++] = val;
        /* If we read a newline or carriage return, it means we've reached the end of a vector. */
        if (res == 2 && (c == '\n' || c == '\r')) {
            if (*n == 0) *d = col;
            curr = append_row(&head, curr, buf, *d);
            if (!curr) { free(buf); free_vector_list(head); return NULL; }
            (*n)++; col = 0;
        }
        if (res == 1) break;
    }
    /* Handle the last line if it doesn't end with a newline */
    if (col > 0) {
        if (*n == 0) *d = col;
        curr = append_row(&head, curr, buf, *d);
        if (curr) (*n)++;
    }
    free(buf);
    return head;
}

/*
 * Reads data from a file and converts it into a 2D matrix.
 * The function first loads the data into a linked list of vectors, then converts it to a matrix.
 * Caller is responsible for freeing the allocated matrix memory.
 */
double** read_file(const char* filename, int* n, int* d) {
    vector* data_list;
    double** data_matrix;
    FILE* f = fopen(filename, "r");
    if (!f) {   
        printf("%s\n", ERROR_MSG);
        return NULL;
    }
    data_list = load_data(f, n, d);
    fclose(f);

    if(!data_list) return NULL;

    data_matrix = vector_to_matrix(data_list, *n, *d);
    return data_matrix;
}

/*
 * Converts a linked list of vectors into a 2D matrix.
 * The function allocates memory for the matrix and copies values from the vector list.
 * It also frees the linked list nodes and their internal data arrays to prevent leaks.
 * Caller is responsible for freeing the allocated matrix memory using free_matrix.
 */
double** vector_to_matrix(vector* head, int n, int d) {
    int i, j;
    vector* temp;
    double** matrix = (double**)malloc(n * sizeof(double*));
    if (!matrix) {
        printf("%s\n", ERROR_MSG);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        /* Allocate space for each row (Deep Copy) using dimension d */
        matrix[i] = (double*)malloc(d * sizeof(double));
        if (!matrix[i]) {
            free_matrix(matrix, i); /* Free already allocated rows if this one fails */
            printf("%s\n", ERROR_MSG);
            return NULL;
        }
        /* Copy data from the current vector node to the matrix row */
        for (j = 0; j < d; j++) {
            matrix[i][j] = head->data[j];
        }
        /* Move to the next vector and clean up the current node's memory */
        temp = head;
        head = head->next;

        /* Free the current node's data array and the node itself to prevent memory leaks. */
        if (temp->data != NULL) {
            free(temp->data);
        }
        free(temp); 
    }
    
    return matrix;
}

/*
 * Calculates the squared Euclidean distance between two vectors.
 */
double sq_distance(double* v1, double* v2, int d) {
    double sum, diff;
    int i;

    sum = 0.0;
    for (i = 0; i < d; i++) {
        diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

/*
 * Calculates the Similarity Matrix based on the input data.
 * The similarity between two vectors is defined as exp(-sq_distance/2).
 * The diagonal of the similarity matrix is set to 0 (no self-similarity).
 * Caller is responsible for freeing the allocated matrix memory.
 */
double** sym(double** data, int n, int d){
    double** sim_matrix;
    int i, j;

    sim_matrix = allocate_matrix(n, n);
    if(!sim_matrix) return NULL;

    for (i = 0; i < n; i++) {
        /* Compute similarity for pairs (i, j) where j > i to avoid redundant calculations */
        for (j = i + 1; j < n; j++) {
            double s = exp(-sq_distance(data[i], data[j], d) / 2.0);
            sim_matrix[i][j] = s;
            sim_matrix[j][i] = s;
        }
        /* Set diagonal to 0 (no self-similarity) */
        sim_matrix[i][i] = 0.0;
    }
    return sim_matrix;
}

/*
 * Calculates the Diagonal Degree Matrix based on the input similarity matrix.
 * The degree of a node is the sum of its similarities to all other nodes.
 * The degree matrix is a diagonal matrix where D[i][i] = degree of node i.
 * Caller is responsible for freeing the allocated matrix memory.
 */
double** ddg(double** data, int n, int d) {
    double **A, **D;
    A = sym(data, n, d);
    if(!A) return NULL;
    D = compute_ddg(A, n);
    free_matrix(A, n);
    return D;
}

/**
 * Internal Engine: Calculates the Diagonal Degree Matrix D from a Similarity Matrix A.
 * D_ii = sum of row i in A.
 */
double** compute_ddg(double** A, int n) {
    int i, j;
    double sum;
    double** degree_matrix = allocate_matrix(n, n);
    if(!degree_matrix) return NULL;

    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        /* Zero initialized by allocate_matrix, only set diagonal */
        degree_matrix[i][i] = sum; 
    }
    return degree_matrix;
}

/*
 * Calculates the Normalized Similarity Matrix.
 * The normalized similarity is defined as D^(-1/2) * A * D^(-1/2), where A is the similarity matrix and D is the degree matrix.
 * Caller is responsible for freeing the allocated matrix memory.
 */
double** norm(double** data, int n, int d){
    double **A, **W, **D;
    double *inv_sqrt_diag;

    A = sym(data, n, d); /* Calculate A once */
    if(!A) return NULL;

    D = compute_ddg(A, n); /* Calculate D from A */
    if(!D) {
        free_matrix(A, n);
        return NULL;
    }

    inv_sqrt_diag = get_inv_sqrt_diag(D, n);
    if(!inv_sqrt_diag) {
        free_matrix(A, n);
        free_matrix(D, n);
        return NULL;
    }

    W = compute_normalized_matrix(A, inv_sqrt_diag, n); /* Compute W = D^(-1/2) * A * D^(-1/2) */

    free_matrix(A, n);
    free_matrix(D, n);
    free(inv_sqrt_diag);
    return W;
}

/*
 * Computes the normalized similarity matrix W using the similarity matrix A and the inverse square root of the degree matrix.
 * W = D^(-1/2) * A * D^(-1/2)
 * Caller is responsible for freeing the allocated matrix memory.
 */
double** compute_normalized_matrix(double** A, double* inv_sqrt_diag, int n) {
    int i, j;
    double** W = allocate_matrix(n, n);
    if(!W) return NULL;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* W_ij = A_ij * (1/sqrt(D_ii)) * (1/sqrt(D_jj)) */
            W[i][j] = inv_sqrt_diag[i] * A[i][j] * inv_sqrt_diag[j];
        }
    }
    return W;
}

/*
 * Multiplies two matrices A and B, storing the result in mul.
 * A is of size n x m, B is of size m x p, and mul is of size n x p.
 */
void multiply(double** A, double** B, double** mul, int n, int m, int p){
    int i, j, k;
    double temp;
    for (i = 0; i < n; i++) {
        /* Initialize the result matrix element to 0 before accumulation */
        for (j = 0; j < p; j++) {
            mul[i][j] = 0.0;
        }
        for (k = 0; k < m; k++) {
            temp = A[i][k];
            if (temp == 0) continue;
            for (j = 0; j < p; j++){
                mul[i][j] += temp * B[k][j];
            }
            
        }
    }
}

/*
 * Transposes a matrix mat of size rows x cols, storing the result in transposed.
 * Caller is responsible for ensuring transposed is allocated with dimensions cols x rows.
 */
void transpose(double** mat, double** transposed, int rows, int cols){
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            transposed[j][i] = mat[i][j];
        }
    }
}

/*
 * Calculates the inverse square root of the diagonal elements of a matrix D.
 * The result is a diagonal matrix where the i-th diagonal element is 1/sqrt(D[i][i]).
 * Caller is responsible for freeing the allocated matrix memory.
 */
double* get_inv_sqrt_diag(double** D, int n){
    int i;
    double* inv_sqrt_diag = (double*)malloc(n * sizeof(double));
    if (!inv_sqrt_diag) {
        printf("%s\n", ERROR_MSG);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        if (D[i][i] > 0) {
            inv_sqrt_diag[i] = 1.0 / sqrt(D[i][i]);
        } else {
            inv_sqrt_diag[i] = 0.0; /* Handle zero degree case */ 
        }
    }
    return inv_sqrt_diag;
}

/*
 * Updates the matrix H based on the current W and H using the multiplicative update rule.
 * This is one iteration of the SymNMF optimization process.
 * Caller is responsible for ensuring W and H are properly allocated and have compatible dimensions.
 */
void update_h(double** H, double** W, double **WH, double **Ht, double **HtH,double **HHtH, int n, int k){
    int i, j;
    double denom, nom;
    double eps_h = 1e-6;

    multiply(W, H, WH, n, n, k); /* W * H */
    transpose(H, Ht, n, k); /* H^T */
    multiply(Ht, H, HtH, k, n, k); /*  H^T * H */
    multiply(H, HtH, HHtH, n, k, k); /* H * (H^T * H) */

    for(i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            denom = HHtH[i][j];
            nom = WH[i][j];
            denom += eps_h; /* Avoid division by zero */
            
            H[i][j] = H[i][j] * (1 - BETA + BETA * (nom / denom));
        }
    }
}

/*
 * The main SymNMF optimization loop.
 * Iteratively updates H based on W until convergence or max iterations.
 * Convergence is determined by the Frobenius norm of the difference between successive H matrices.
 * Caller is responsible for ensuring W and H are properly allocated and have compatible dimensions.
 */
void symnmf(double** H, double** W, int n, int k){
    int iter;
    double diff;
    double** WH = allocate_matrix(n, k);
    double** Ht = allocate_matrix(k, n);
    double** HtH = allocate_matrix(k, k);
    double** HHtH = allocate_matrix(n, k);
    double **H_old = allocate_matrix(n, k);

    if(!H_old || !WH || !Ht || !HtH || !HHtH) return;

    for (iter = 0; iter < MAX_ITER; iter++) {
        copy_matrix_values(H_old, H, n, k); /* Keep a copy of the old H for convergence check */

        update_h(H, W, WH, Ht, HtH, HHtH, n, k);
        diff = frobenius_sq_diff(H, H_old, n, k);

        if (diff < EPSILON) {
            break;
        }
    }
    free_matrix(H_old, n);
    free_matrix(WH, n);
    free_matrix(Ht, k);
    free_matrix(HtH, k);
    free_matrix(HHtH, n);
}

/*
 * Calculates the squared Frobenius norm of the difference between two matrices H and H_new.
 * Caller is responsible for ensuring the matrices have compatible dimensions.
 */
double frobenius_sq_diff(double** H, double** H_new, int n, int k){
    int i, j;
    double diff, sum = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = H[i][j] - H_new[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/*
 * Prints a matrix to the standard output.
 * mat is of size rows x cols.
 */
void print_matrix(double** mat, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", mat[i][j]);
            if (j < cols - 1) printf(","); 
        }
        printf("\n");
    }
}

/*
 * Helper function to route the requested goal and execute the corresponding math.
 * Returns the resulting matrix, or NULL if the goal is invalid.
 */
double** execute_goal(char *goal, double **data, int n, int d) {
    if (strcmp(goal, SYM) == 0) return sym(data, n, d);
    if (strcmp(goal, DDG) == 0) return ddg(data, n, d);
    if (strcmp(goal, NORM) == 0) return norm(data, n, d);
    
    /* Invalid goal */
    printf("%s\n", ERROR_MSG);
    return NULL;
}

/*
 * Main function to execute the SymNMF algorithm.
 * Parses command line arguments, reads input data, and routes to the appropriate goal function.
 */
int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **data, **result;
    int n, d;

    /* Validate number of arguments */
    if (argc != 3) {
        printf("%s\n", ERROR_MSG);
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    /* Read input file and get dimensions */
    data = read_file(file_name, &n, &d);
    if (!data) return 1;

    /* Execute the requested mathematical goal */
    result = execute_goal(goal, data, n, d);
    if (!result) {
        free_matrix(data, n);
        return 1;
    }

    /* Print the resulting matrix and free memory */
    print_matrix(result, n, n);
    free_matrix(result, n);
    free_matrix(data, n);
    
    return 0;
}