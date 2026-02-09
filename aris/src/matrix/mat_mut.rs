use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use super::{
    ColIter, ColIterMut, DiagIter, MatEnumerate, MatMut, MatRef, RowIter, RowIterMut, fmt_matrix,
    fmt_matrix_debug,
};

impl<'a, T> MatMut<'a, T> {
    #[inline(always)]
    pub(crate) fn ptr_at(&self, i: usize, j: usize) -> *const T {
        unsafe {
            (self.ptr as *const T)
                .offset(i as isize * self.row_stride + j as isize * self.col_stride)
        }
    }

    #[inline(always)]
    pub(crate) fn ptr_at_mut(&mut self, i: usize, j: usize) -> *mut T {
        unsafe {
            self.ptr
                .offset(i as isize * self.row_stride + j as isize * self.col_stride)
        }
    }

    #[inline(always)]
    pub(crate) fn at(&self, i: usize, j: usize) -> &T {
        unsafe { &*self.ptr_at(i, j) }
    }

    #[inline(always)]
    pub(crate) fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        unsafe { &mut *self.ptr_at_mut(i, j) }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.nrows * self.ncols
    }

    #[inline]
    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    #[inline]
    pub fn rb(&self) -> MatRef<'_, T> {
        MatRef {
            ptr: self.ptr as *const T,
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn rb_mut(&mut self) -> MatMut<'_, T> {
        MatMut {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.nrows && j < self.ncols {
            Some(self.at(i, j))
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < self.nrows && j < self.ncols {
            Some(self.at_mut(i, j))
        } else {
            None
        }
    }

    pub fn col_iter(&self) -> ColIter<'_, T> {
        self.rb().col_iter()
    }

    pub fn row_iter(&self) -> RowIter<'_, T> {
        self.rb().row_iter()
    }

    pub fn diag_iter(&self) -> DiagIter<'_, T> {
        self.rb().diag_iter()
    }

    pub fn enumerate(&self) -> MatEnumerate<'_, T> {
        self.rb().enumerate()
    }

    pub fn col_iter_mut(self) -> ColIterMut<'a, T> {
        ColIterMut {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            col: 0,
            _marker: PhantomData,
        }
    }

    pub fn row_iter_mut(self) -> RowIterMut<'a, T> {
        RowIterMut {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            row: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> Index<(usize, usize)> for MatMut<'_, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({}, {}) out of bounds for {}x{} matrix",
            i,
            j,
            self.nrows,
            self.ncols
        );
        self.at(i, j)
    }
}

impl<T> IndexMut<(usize, usize)> for MatMut<'_, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({}, {}) out of bounds for {}x{} matrix",
            i,
            j,
            self.nrows,
            self.ncols
        );
        self.at_mut(i, j)
    }
}

impl<T: fmt::Display> fmt::Display for MatMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(self.rb(), f)
    }
}

impl<T: fmt::Debug> fmt::Debug for MatMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix_debug("MatMut", self.rb(), f)
    }
}

impl<'a, T> Iterator for ColIterMut<'a, T> {
    type Item = MatMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col >= self.ncols {
            return None;
        }
        let col_ptr = unsafe { self.ptr.offset(self.col as isize * self.col_stride) };
        let col_view = MatMut {
            ptr: col_ptr,
            nrows: self.nrows,
            ncols: 1,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        self.col += 1;
        Some(col_view)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ncols - self.col;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for ColIterMut<'_, T> {}

impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = MatMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.nrows {
            return None;
        }
        let row_ptr = unsafe { self.ptr.offset(self.row as isize * self.row_stride) };
        let row_view = MatMut {
            ptr: row_ptr,
            nrows: 1,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        self.row += 1;
        Some(row_view)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.nrows - self.row;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for RowIterMut<'_, T> {}
