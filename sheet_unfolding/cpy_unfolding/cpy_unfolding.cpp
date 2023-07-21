#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cstdint> 

#include "unfolding.h"

void cleanup_references(int numArgs, ...)
{
    va_list args;
    va_start(args, numArgs);

    for (int i = 0; i < numArgs; ++i)
    {
        PyObject* obj = va_arg(args, PyObject*);
        Py_XDECREF(obj);
    }

    va_end(args);
}

// Function to create an array similar to np.arange
PyArrayObject* create_range_array(int64_t start, int64_t stop) {
    // Calculate the number of elements in the array
    npy_intp size = stop - start;

    // Create the array with the desired shape and data type
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_INT64);

    // Fill the array with the range of values
    int64_t* data = (int64_t*)PyArray_DATA(array);
    for (npy_intp i = 0; i < size; i++) {
        data[i] = start + i;
    }

    return array;
}

// Wrapper function to be called from Python
static PyObject* cpy_unfold(PyObject* self, PyObject* args)
{
    PyObject *pos_obj, *mass_obj, *simplices_obj, *idptr_obj = nullptr;
    
    float boxsize;
    
    int mode = 2, dimension=2;

    // Parse the input NumPy array from Python
    if (!PyArg_ParseTuple(args, "ifOOO|O",  &mode, &boxsize, &pos_obj, &mass_obj, &simplices_obj, &idptr_obj))
        return nullptr;

    if (mode >= 3)
        dimension = 3;
    
    
    printf("dimension %d\n", dimension);
    fflush(stdout);
    // Convert the input object to a NumPy array
    
    PyArrayObject *inpos_array, *inmass_array, *insimplices_array, *inidptr_array = nullptr;
    PyArrayObject *pos_array, *mass_array, *simplices_array, *idptr_array;
    
    
    inpos_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(pos_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY));
    inmass_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mass_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY));
    insimplices_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(simplices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    
    pos_array = (PyArrayObject*)PyArray_NewCopy(inpos_array, NPY_CORDER);
    mass_array = (PyArrayObject*)PyArray_NewCopy(inmass_array, NPY_CORDER);
    simplices_array = (PyArrayObject*)PyArray_NewCopy(insimplices_array, NPY_CORDER);
    
    // Check if the array is of type float32 (single-precision float)
    int64_t npart = PyArray_SIZE(mass_array);
    int64_t simp_size = PyArray_SIZE(simplices_array);
    
    if(idptr_obj == nullptr)
    {
        idptr_array = create_range_array(0, npart);
    }
    else
    {
        inidptr_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(idptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
        idptr_array = (PyArrayObject*)PyArray_NewCopy(inidptr_array, NPY_CORDER);
    }
    
    cleanup_references(4, inpos_array, inmass_array, insimplices_array, inidptr_array);

    if ((PyArray_SIZE(pos_array) != dimension*npart) || (PyArray_SIZE(idptr_array) != npart)) {
        PyErr_SetString(PyExc_ValueError, "Input shapes do not match");
        cleanup_references(4, pos_array, mass_array, simplices_array, idptr_array);
        return nullptr;
    }

    float* data_pos = static_cast<float*>(PyArray_DATA(pos_array));
    float* data_mass = static_cast<float*>(PyArray_DATA(mass_array));
    npy_int64* data_simplices = static_cast<npy_int64*>(PyArray_DATA(simplices_array));
    npy_int64* data_idptr = static_cast<npy_int64*>(PyArray_DATA(idptr_array));

    if(mode == 1)
        unfold2d_sorted(boxsize, npart, simp_size/3, data_pos, data_mass, data_idptr, data_simplices);
    else if(mode == 2)
        unfold2d(boxsize, npart, simp_size/3, data_pos, data_mass, data_idptr, data_simplices);
    else if(mode == 3)
        unfold3d(boxsize, npart, simp_size/4, data_pos, data_mass, data_idptr, data_simplices);
    else if(mode >= 4)
        unfold3d_sorted(boxsize, npart, simp_size/4, data_pos, data_mass, data_idptr, data_simplices, mode);
    
    //cleanup_references(4, inpos_array, inmass_array, simplices_array, idptr_array);
    
    // Pack the arrays into a tuple
    PyObject* result_tuple = PyTuple_New(4);
    PyTuple_SetItem(result_tuple, 0, (PyObject*)pos_array);
    PyTuple_SetItem(result_tuple, 1, (PyObject*)mass_array);
    PyTuple_SetItem(result_tuple, 2, (PyObject*)simplices_array);
    PyTuple_SetItem(result_tuple, 3, (PyObject*)idptr_array);

    return result_tuple;
}

// Method definition
static PyMethodDef module_methods[] = {
    {"cpy_unfold", cpy_unfold, METH_VARARGS, "Unfold the Sheet"},
    {nullptr, nullptr, 0, nullptr}
};



// Module definition
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "cpy_unfolding",
    "Module description",
    -1,
    module_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_cpy_unfolding(void)
{
    import_array();  // Initialize NumPy API
    return PyModule_Create(&module_definition);
}