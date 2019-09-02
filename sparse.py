# sparse.py

from __future__ import print_function

"""
Utility classes and functions.
"""

try:
    xrange
except NameError:
    xrange = range
import os
import numpy as np

from . import cext

class Sparse (object):

    def __init__ (self, init, shape, _check=True):
        data, (row, column) = init
        if _check:
            data, row, column = map (np.atleast_1d, (data, row, column))
        self.data, self.row, self.column = data, row, column
        self.shape = self.n_row, self.n_column = shape
        self.dtype = data.dtype
        self.ids = self._ij_to_id (row, column)
        if _check:
            order = np.argsort (self.ids)
            self.data, self.row, self.column = data[order], row[order], column[order]
            self.ids = self.ids[order]

    def toarray (self):
        out = np.zeros (self.shape, dtype=self.dtype)
        out[self.row, self.column] = self.data
        return out

    def _ij_to_id (self, i, j):
        return i * self.n_column + j

    def _id_to_ij (self, ids):
        return ids / self.n_column, ids % self.n_column

    def nonzero (self):
        mask_sparse = (self != 0)
        mask = mask_sparse.data
        return self.row[mask], self.column[mask]

    def broadcast1d (self, f, *a):
        data = f (self.data, *a)
        return type (self) ((data, (self.row, self.column)), self.shape)

    def __ge__ (self, value):
        return self.broadcast1d (np.greater_equal, value)
    def __gt__ (self, value):
        return self.broadcast1d (np.greater, value)

    def __le__ (self, value):
        return self.broadcast1d (np.less_equal, value)
    def __lt__ (self, value):
        return self.broadcast1d (np.less, value)

    def __eq__ (self, value):
        return self.broadcast1d (np.equal, value)
    def __ne__ (self, value):
        return self.broadcast1d (np.not_equal, value)

    def broadcast2d (a, b, f, afill=None, bfill=None):
        # shapes must match
        assert a.shape == b.shape
        # fill both or neither
        assert (int (afill is None) + int (bfill is None)) % 2 == 0
        if a.data.size == b.data.size and np.all (a.ids == b.ids):
            data = f (a.data, b.data)
            return type (a) ((data, (a.row, a.column)), a.shape)
        intersect = (afill is None)
        both_ids = np.intersect1d (a.ids, b.ids)
        a_ids = np.intersect1d (a.ids, both_ids)
        b_ids = np.intersect1d (b.ids, both_ids)
        i1 = np.searchsorted (both_ids, a_ids)
        i2 = np.searchsorted (both_ids, b_ids)
        data = f (a.data[i1], b.data[i2])
        if intersect:
            row, column = a._id_to_ij (both_ids)
        else:
            only_a_ids = np.setdiff1d (a.ids, both_ids)
            only_b_ids = np.setdiff1d (b.ids, both_ids)
            all_ids = np.union1d (a.ids, b.ids)
            orig_data = data
            data = np.empty (len (all_ids), dtype=orig_data.dtype)
            i_both = np.searchsorted (all_ids, both_ids)
            i1_in = np.searchsorted (a.ids, only_a_ids)
            i2_in = np.searchsorted (b.ids, only_b_ids)
            i1_out = np.searchsorted (all_ids, only_a_ids)
            i2_out = np.searchsorted (all_ids, only_b_ids)
            data[i_both] = orig_data
            data[i1_out] = afill (a.data[i1_in])
            data[i2_out] = afill (a.data[i2_in])
            row, column = a._id_to_ij (all_ids)
        data = np.atleast_1d (data)
        return type (a) ((data, (row, column)), a.shape)

    def add (a, b):
        fill = lambda x: x
        return a.broadcast2d (b, np.add, afill=fill, bfill=fill)
    def subtract (a, b):
        afill = lambda x: x
        bfill = lambda x: -x
        return a.broadcast2d (b, np.subtract, afill=afill, bfill=bfill)
    def multiply (a, b):
        return a.broadcast2d (b, np.multiply)
    def divide (a, b):
        return a.broadcast2d (b, np.divide)

    def dot_sparse (self, v):
        assert v.shape == (self.n_column,)
        data = self.data
        product = data * v[self.column]
        row_bounds = np.r_[0, np.where (np.diff (self.row) >= 1)[0], data.size - 1]
        cumsum = np.r_[0, np.cumsum (product)[1:]]
        data = np.diff (cumsum[row_bounds])
        row = self.row[row_bounds[1:]]
        column = np.zeros (data.size, dtype=int)
        return type (self) ((data, (row, column)), shape=(self.n_row, 1))
        
    def dot_py (self, v):
        assert v.shape == (self.n_column,)
        data = self.data
        product = data * v[self.column]
        row_bounds = np.r_[0, np.where (np.diff (self.row) >= 1)[0], data.size - 1]
        cumsum = np.r_[0, np.cumsum (product)[1:]]
        values = np.diff (cumsum[row_bounds])
        row = self.row[row_bounds[1:]]
        out = np.zeros (self.n_row)
        out[row] = values
        return out

    def dot (self, v):
        return cext.sparse_dot (self.data, self.row, self.column, v, self.n_row)

    def _normalize_indices (self, x, xmax):
        if isinstance (x, slice):
            x = np.arange (x.start or 0, x.stop or xmax)
        else:
            x = np.atleast_1d (x)
            if x.dtype == np.dtype (bool):
                x = np.arange (xmax)[x]
        if not (np.max (x) < xmax):
            raise ValueError (
                'index {} too large for axis with length {}'.format (
                    np.min (x[x >= xmax]), xmax))
        return x
    def _get_columns (self, j):
        orig_j = j = self._normalize_indices (j, self.n_column)
        assert np.all (np.diff (j) > 0)
        idx = np.in1d (self.ids % self.n_column, j)
        data = self.data[idx]
        row = self.row[idx]
        self_col_idx = self.column[idx]
        relevant_columns = np.unique (np.r_[self_col_idx, j[~np.in1d (j, self_col_idx)]])
        column = np.searchsorted (relevant_columns, self_col_idx)
        return type (self) ((data, (row, column)), shape=(len (j), self.n_column))
    def _get_rows (self, i):
        orig_i = i = self._normalize_indices (i, self.n_row)
        assert np.all (np.diff (i) > 0)

        self_row = self.row
        idx = self.fast_in1d (self_row, i)
        data = self.data[idx]
        column = self.column[idx]
        self_row_idx = self.row[idx]
        relevant_rows = np.unique (np.r_[self_row_idx, i[~self.fast_in1d (i, self_row_idx)]])
        row = np.searchsorted (relevant_rows, self_row_idx)
        return type (self) ((data, (row, column)), shape=(len (i), self.n_column), _check=False)
    def _get_elements (self, i, j):
        orig_i = i = self._normalize_indices (i, self.n_row)
        orig_j = j = self._normalize_indices (j, self.n_column)
        ids = np.intersect1d (self._ij_to_id (i, j), self.ids)
        idx = np.searchsorted (self.ids, ids)
        data = self.data[idx]
        row, column = self._id_to_ij (ids)
        return type (self) ((data, (row, column)), shape=self.shape)

    def __getitem__ (self, ij):
        if isinstance (ij, tuple):
            if len (ij) == 1:
                i, j = ij, slice (None)
            elif len (ij) == 2:
                i, j = ij
            else:
                raise TypeError (
                    '{} is an invalid number of arguments for getitem'.format (len (ij)))
        else:
            i, j = ij, slice (None)
        if isinstance (i, slice) and i == slice (None):
            return self._get_columns (j)
        elif isinstance (j, slice) and j == slice (None):
            return self._get_rows (i)
        else:
            return self._get_elements (i, j)

    @staticmethod
    def fast_in1d (a_test, a_bins):
        bins = np.transpose ([a_bins - .25, a_bins + .25]).ravel ()
        return np.searchsorted (bins, a_test) % 2 > 0


