from __future__ import division, print_function, absolute_import

import sys
import numpy as np
import scipy.sparse

# from scipy._lib._version import NumpyVersion

__all__ = ['save_sparse_npz', 'load_sparse_npz']


if np.__version__ >= '1.10.0':
    # Make loading safe vs. malicious input
    PICKLE_KWARGS = dict(allow_pickle=False)
else:
    PICKLE_KWARGS = dict()


def save_sparse_npz(file, matrix, compressed=True):
    """ Save a sparse matrix to a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already
        there.
    matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
        The sparse matrix to save.
    compressed : bool, optional
        Allow compressing the file. Default: True

    See Also
    --------
    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
    numpy.savez: Save several arrays into a ``.npz`` archive.
    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import scipy.sparse
    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)

    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)
    """

    arrays_dict = dict(format=matrix.format.encode('ascii'),
                       shape=matrix.shape,
                       data=matrix.data)
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        raise NotImplementedError('Save is not implemented for sparse matrix of format {}.'.format(matrix.format))

    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_sparse_npz(file):
    """ Load a sparse matrix from a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be loaded.

    Returns
    -------
    result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
        A sparse matrix containing the loaded data.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.

    See Also
    --------
    scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.
    numpy.load: Load several arrays from a ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import scipy.sparse
    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)

    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
        with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)
    """

    with np.load(file, **PICKLE_KWARGS) as loaded:
        try:
            matrix_format = loaded['format']
        except KeyError:
            raise ValueError('The file {} does not contain a sparse matrix.'.format(file))

        matrix_format = matrix_format.item()

        if sys.version_info[0] >= 3 and not isinstance(matrix_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with Scipy < 1.0.0 may contain unicode or bytes.
            matrix_format = matrix_format.decode('ascii')

        try:
            cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
        except AttributeError:
            raise ValueError('Unknown matrix format "{}"'.format(matrix_format))

        if matrix_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        elif matrix_format == 'dia':
            return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
        elif matrix_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
        else:
            raise NotImplementedError('Load is not implemented for '
                                      'sparse matrix of format {}.'.format(matrix_format))



def reshape_sparse(a, shape,ReturnType = 'coo'):
    """Reshape the sparse matrix `a`.
    https://stackoverflow.com/questions/16511879/reshape-sparse-matrix-efficiently-python-scipy-0-12
    Returns a coo_matrix with shape `shape`.
    can return ReturnType sparse matrix
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = scipy.sparse.coo_matrix((c.data, (new_row, new_col)), shape=shape)

    if ReturnType.upper() == 'CSR':
        b = scipy.sparse.csr_matrix(b)
    elif ReturnType.upper() == 'CSC':
        b = scipy.sparse.csc_matrix(b)
    return b