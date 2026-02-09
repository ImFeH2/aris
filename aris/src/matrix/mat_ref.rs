use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;

use super::{ColIter, DiagIter, Mat, MatEnumerate, MatRef, RowIter, fmt_matrix, fmt_matrix_debug};

impl<'a, T> MatRef<'a, T> {
    #[inline(always)]
    pub(crate) fn ptr_at(self, i: usize, j: usize) -> *const T {
        unsafe {
            self.ptr
                .offset(i as isize * self.row_stride + j as isize * self.col_stride)
        }
    }

    #[inline(always)]
    pub(crate) fn at(self, i: usize, j: usize) -> &'a T {
        unsafe { &*self.ptr_at(i, j) }
    }

    #[inline]
    pub fn nrows(self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn shape(self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    #[inline]
    pub fn size(self) -> usize {
        self.nrows * self.ncols
    }

    #[inline]
    pub fn row_stride(self) -> isize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(self) -> isize {
        self.col_stride
    }

    #[inline]
    pub fn get(self, i: usize, j: usize) -> Option<&'a T> {
        if i < self.nrows && j < self.ncols {
            Some(self.at(i, j))
        } else {
            None
        }
    }

    pub fn to_owned(self) -> Mat<T>
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(self.nrows * self.ncols);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(self.at(i, j).clone());
            }
        }
        Mat::from_vec_col(self.nrows, self.ncols, data)
    }

    pub fn col_iter(self) -> ColIter<'a, T> {
        ColIter {
            matrix: self,
            col: 0,
        }
    }

    pub fn row_iter(self) -> RowIter<'a, T> {
        RowIter {
            matrix: self,
            row: 0,
        }
    }

    pub fn diag_iter(self) -> DiagIter<'a, T> {
        let length = self.nrows.min(self.ncols);
        DiagIter {
            matrix: self,
            index: 0,
            length,
        }
    }

    pub fn enumerate(self) -> MatEnumerate<'a, T> {
        MatEnumerate {
            matrix: self,
            index: 0,
            length: self.nrows * self.ncols,
        }
    }
}

impl<T: PartialEq> PartialEq for MatRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return false;
        }
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if self.at(i, j) != other.at(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Eq> Eq for MatRef<'_, T> {}

impl<T> Index<(usize, usize)> for MatRef<'_, T> {
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

impl<T: fmt::Display> fmt::Display for MatRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(*self, f)
    }
}

impl<T: fmt::Debug> fmt::Debug for MatRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix_debug("MatRef", *self, f)
    }
}

impl<'a, T> Iterator for ColIter<'a, T> {
    type Item = MatRef<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col >= self.matrix.ncols {
            return None;
        }
        let col_ref = MatRef {
            ptr: self.matrix.ptr_at(0, self.col),
            nrows: self.matrix.nrows,
            ncols: 1,
            row_stride: self.matrix.row_stride,
            col_stride: self.matrix.col_stride,
            _marker: PhantomData,
        };
        self.col += 1;
        Some(col_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.ncols - self.col;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for ColIter<'_, T> {}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = MatRef<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.matrix.nrows {
            return None;
        }
        let row_ref = MatRef {
            ptr: self.matrix.ptr_at(self.row, 0),
            nrows: 1,
            ncols: self.matrix.ncols,
            row_stride: self.matrix.row_stride,
            col_stride: self.matrix.col_stride,
            _marker: PhantomData,
        };
        self.row += 1;
        Some(row_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.nrows - self.row;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for RowIter<'_, T> {}

impl<'a, T> Iterator for DiagIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            return None;
        }
        let item = self.matrix.at(self.index, self.index);
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for DiagIter<'_, T> {}

impl<'a, T> Iterator for MatEnumerate<'a, T> {
    type Item = ((usize, usize), &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            return None;
        }
        let ncols = self.matrix.ncols;
        if ncols == 0 {
            return None;
        }
        let i = self.index / ncols;
        let j = self.index % ncols;
        self.index += 1;
        Some(((i, j), self.matrix.at(i, j)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for MatEnumerate<'_, T> {}
