#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ERROR_MSG "An Error Has Occurred"
#define SYMNMF "symnmf"
#define SYM "sym"
#define DDG "ddg"
#define NORM "norm"
#define BETA 0.5
#define MAX_ITER 300
#define EPSILON 1e-4

/*
 * Vector structure for storing data points
 */
typedef struct vector {
    double* data;
    struct vector* next;
} vector;


/* Memory Management */
double** allocate_matrix(int rows, int cols);
void free_matrix(double** mat, int rows);
void copy_matrix_values(double** dest, double** src, int n, int k);
void free_vector_list(vector* head);

/* File Parsing */
vector* load_data(FILE* f, int* n, int* d);
double* resize_buffer(double* buffer, int* capacity);
vector* create_node(double* temp_row, int d);
double** vector_to_matrix(vector* head, int n, int d);
double** read_file(const char* filename, int* n, int* d);
vector* append_row(vector **head, vector *curr, double *buf, int d);

/* Core Mathematical Engine */
double sq_distance(double* v1, double* v2, int d);
double** sym(double** data, int n, int d);
double** ddg(double** data, int n, int d);
double** compute_ddg(double** A, int n);
double** norm(double** data, int n, int d);

/* Matrix Operations (for SymNMF/Update) */
double** compute_normalized_matrix(double** A, double* inv_sqrt_diag, int n);
void multiply(double** A, double** B, double** mul, int n, int m, int p);
void transpose(double** mat, double** transposed, int rows, int cols);
double* get_inv_sqrt_diag(double** D, int n);

/* SymNMF Optimization */
void update_h(double** H, double** W, double **WH, double **Ht, double **HtH,double **HHtH, int n, int k);
void symnmf(double** H, double** W, int n, int k);
double frobenius_sq_diff(double** H, double** H_new, int n, int k);

/* Output */
void print_matrix(double** mat, int rows, int cols);
double** execute_goal(char *goal, double **data, int n, int d);

#endif