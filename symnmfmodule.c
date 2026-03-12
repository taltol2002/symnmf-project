#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Converts Python List of Lists to C double** */
static double** python_to_c_array(PyObject* list_obj, int n, int d);
/* Converts C double** back to Python List of Lists */
static PyObject* c_array_to_python(double** matrix, int n, int d);

/* Generic helper to execute sym, ddg, or norm without code duplication */
static PyObject* execute_matrix_goal(PyObject* args, double** (*goal_func)(double**, int, int));
/* Wraps the sym function */
static PyObject* sym_wrapper(PyObject* self, PyObject* args);
/* Wraps the ddg function */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args);
/* Wraps the norm function */
static PyObject* norm_wrapper(PyObject* self, PyObject* args);
/* Wraps the full iterative symnmf (receives W and H) */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args);
/* The initialization function */
PyMODINIT_FUNC PyInit_symnmfmodule(void);


/* The methods table: maps Python names to C functions */
static PyMethodDef symnmf_methods[] = {
    {
        "sym",                   /* The name as seen from Python */
        (PyCFunction)sym_wrapper, /* Point to the WRAPPER, not the logic! */
        METH_VARARGS,            /* Arguments are passed as a tuple */
        PyDoc_STR("Similarity Matrix calculation") /* Docstring */
    },
    {
        "ddg", 
        (PyCFunction)ddg_wrapper, 
        METH_VARARGS, 
        PyDoc_STR("Diagonal Degree Matrix calculation")
    },
    {
        "norm", 
        (PyCFunction)norm_wrapper, 
        METH_VARARGS, 
        PyDoc_STR("Normalized Similarity Matrix calculation")
    },
    {
        "symnmf", 
        (PyCFunction)symnmf_wrapper, 
        METH_VARARGS, 
        PyDoc_STR("Full SymNMF optimization loop")
    },
    {NULL, NULL, 0, NULL} /* Sentinel - marks the end of the array */
};

/* The module definition structure */
static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",          /* name of module as seen in Python */
    NULL,              /* module documentation, may be NULL */
    -1,                /* size of per-interpreter state of the module, or -1 */
    symnmf_methods     /* the methods table */
};


/**
 * Converts a Python list of lists into a C double** matrix.
 * Assumes the input is a valid list of lists of floats.
 */
static double** python_to_c_array(PyObject* list_obj, int n, int d) {
    int i, j;
    PyObject* row;
    double** matrix = allocate_matrix(n, d); /* Using your C allocator */
    if (!matrix) return NULL;
    for (i = 0; i < n; i++) {
        /* Get the i-th row from the outer Python list */
        row = PyList_GetItem(list_obj, i);
        for (j = 0; j < d; j++) {
            /* Convert each Python float to a C double */
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return matrix;
}

/**
 * Converts a C double** matrix back into a Python list of lists.
 * This allows returning the results to the Python script.
 */
static PyObject* c_array_to_python(double** matrix, int n, int d) {
    int i, j;
    PyObject* inner_list;
    PyObject* outer_list = PyList_New(n); /* Create the main list */
    if (!outer_list) return NULL;
    for (i = 0; i < n; i++) {
        inner_list = PyList_New(d);
        if (!inner_list) {
            Py_DECREF(outer_list);
            return NULL;
        }
        for (j = 0; j < d; j++) {
            /* PyList_SetItem "steals" the reference - no extra cleanup needed */
            PyList_SetItem(inner_list, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(outer_list, i, inner_list);
    }
    return outer_list;
}

/* Generic helper to execute sym, ddg, or norm without code duplication */
static PyObject* execute_matrix_goal(PyObject* args, double** (*goal_func)(double**, int, int)) {
    PyObject *input_list, *output_list;
    int n, d;
    double **data, **result;

    /* Parse ONLY ONE argument: the input list (O) */
    if (!PyArg_ParseTuple(args, "O", &input_list)) return NULL;
    
    /* Calculate dimensions inside C */
    n = PyList_Size(input_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, ERROR_MSG);
        return NULL;
    }
    d = PyList_Size(PyList_GetItem(input_list, 0)); 
    if (d == 0) {
        PyErr_SetString(PyExc_ValueError, ERROR_MSG);
        return NULL;
    }

    data = python_to_c_array(input_list, n, d);
    if (!data) {
        PyErr_SetString(PyExc_MemoryError, ERROR_MSG);
        return NULL;
    }

    result = goal_func(data, n, d); 
    if (!result) {
        free_matrix(data, n);
        PyErr_SetString(PyExc_RuntimeError, ERROR_MSG);
        return NULL;
    }

    output_list = c_array_to_python(result, n, n);
    free_matrix(data, n);
    free_matrix(result, n);
    return output_list;
}


/**
 * Wrapper for the Similarity Matrix (sym) calculation.
 * Python call: symnmf.sym(list_of_lists, n, d).
 */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    return execute_matrix_goal(args, sym);
}

/**
 * Wrapper for the Diagonal Degree Matrix (ddg) calculation.
 * Python: symnmf.ddg(data, n, d).
 */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    return execute_matrix_goal(args, ddg);
}

/**
 * Wrapper for the Normalized Similarity Matrix (norm) calculation.
 * Python: symnmf.norm(data, n, d).
 */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    return execute_matrix_goal(args, norm);
}

/**
 * Wrapper for the full SymNMF iterative optimization.
 * Python: symnmf.symnmf(H, W, n, k) where H and W are list of lists.
 * This function expects H and W to be pre-allocated matrices of the correct size.
 * It will call the symnmf function and return the final H matrix as a Python list of lists.
 * The caller is responsible for ensuring that H and W are initialized properly before calling this function.
 * The function will return the final H matrix after optimization.
 * The W matrix is expected to be the normalized similarity matrix (D^(-1/2) * A * D^(-1/2)) that is computed separately and passed in.
 */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject *h_obj, *w_obj, *final_h;
    int n, k;
    double **H, **W;

    /* Parse arguments: H matrix first, then W matrix, n, k */
    if (!PyArg_ParseTuple(args, "OOii", &h_obj, &w_obj, &n, &k)) {
        return NULL;
    }
    if (n == 0 || k == 0) {
        PyErr_SetString(PyExc_ValueError, ERROR_MSG);
        return NULL;
    }

    /* Convert H first */
    H = python_to_c_array(h_obj, n, k);
    if (!H) {
        PyErr_SetString(PyExc_MemoryError, ERROR_MSG);
        return NULL;
    }
    
    /* Convert W second */
    W = python_to_c_array(w_obj, n, n);
    if (!W) {
        PyErr_SetString(PyExc_MemoryError, ERROR_MSG);
        free_matrix(H, n); 
        return NULL;
    }

    /* Call C function exactly as declared: H then W */
    symnmf(H, W, n, k); 
    
    final_h = c_array_to_python(H, n, k);

    free_matrix(H, n);
    free_matrix(W, n);
    return final_h;
}

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    /* Creates the module based on the definition above */
    return PyModule_Create(&symnmf_module);
}