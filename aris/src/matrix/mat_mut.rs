use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use super::{
    ColIter, ColIterMut, DiagIter, Mat, MatEnumerate, MatMut, MatRef, RowIter, RowIterMut,
    fmt_matrix, fmt_matrix_debug,
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

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rb().is_empty()
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        self.rb().is_square()
    }

    #[inline]
    pub fn is_row_vector(&self) -> bool {
        self.rb().is_row_vector()
    }

    #[inline]
    pub fn is_col_vector(&self) -> bool {
        self.rb().is_col_vector()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.rb().is_scalar()
    }

    #[inline]
    pub fn row(&self, i: usize) -> MatRef<'_, T> {
        self.rb().row(i)
    }

    #[inline]
    pub fn col(&self, j: usize) -> MatRef<'_, T> {
        self.rb().col(j)
    }

    #[inline]
    pub fn diagonal(&self) -> MatRef<'_, T> {
        self.rb().diagonal()
    }

    #[inline]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, T> {
        self.rb().submatrix(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn rows_range(&self, range: Range<usize>) -> MatRef<'_, T> {
        self.rb().rows_range(range)
    }

    #[inline]
    pub fn cols_range(&self, range: Range<usize>) -> MatRef<'_, T> {
        self.rb().cols_range(range)
    }

    #[inline]
    pub fn split_at_row(&self, i: usize) -> (MatRef<'_, T>, MatRef<'_, T>) {
        self.rb().split_at_row(i)
    }

    #[inline]
    pub fn split_at_col(&self, j: usize) -> (MatRef<'_, T>, MatRef<'_, T>) {
        self.rb().split_at_col(j)
    }

    #[inline]
    pub fn transpose(&self) -> MatRef<'_, T> {
        self.rb().transpose()
    }

    #[inline]
    pub fn reverse_rows(&self) -> MatRef<'_, T> {
        self.rb().reverse_rows()
    }

    #[inline]
    pub fn reverse_cols(&self) -> MatRef<'_, T> {
        self.rb().reverse_cols()
    }

    #[inline]
    pub fn row_mut(self, i: usize) -> MatMut<'a, T> {
        assert!(
            i < self.nrows,
            "Row index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        MatMut {
            ptr: unsafe { self.ptr.offset(i as isize * self.row_stride) },
            nrows: 1,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn col_mut(self, j: usize) -> MatMut<'a, T> {
        assert!(
            j < self.ncols,
            "Column index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        MatMut {
            ptr: unsafe { self.ptr.offset(j as isize * self.col_stride) },
            nrows: self.nrows,
            ncols: 1,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn diagonal_mut(self) -> MatMut<'a, T> {
        let len = self.nrows.min(self.ncols);
        MatMut {
            ptr: self.ptr,
            nrows: len,
            ncols: 1,
            row_stride: self.row_stride + self.col_stride,
            col_stride: self.row_stride + self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn submatrix_mut(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'a, T> {
        assert!(
            row_start + nrows <= self.nrows,
            "Row range {}..{} out of bounds for {} rows",
            row_start,
            row_start + nrows,
            self.nrows
        );
        assert!(
            col_start + ncols <= self.ncols,
            "Column range {}..{} out of bounds for {} columns",
            col_start,
            col_start + ncols,
            self.ncols
        );
        MatMut {
            ptr: unsafe {
                self.ptr.offset(
                    row_start as isize * self.row_stride + col_start as isize * self.col_stride,
                )
            },
            nrows,
            ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn rows_range_mut(self, range: Range<usize>) -> MatMut<'a, T> {
        assert!(
            range.start <= range.end,
            "Invalid range: start {} > end {}",
            range.start,
            range.end
        );
        let ncols = self.ncols;
        self.submatrix_mut(range.start, 0, range.end - range.start, ncols)
    }

    #[inline]
    pub fn cols_range_mut(self, range: Range<usize>) -> MatMut<'a, T> {
        assert!(
            range.start <= range.end,
            "Invalid range: start {} > end {}",
            range.start,
            range.end
        );
        let nrows = self.nrows;
        self.submatrix_mut(0, range.start, nrows, range.end - range.start)
    }

    pub fn split_at_row_mut(self, i: usize) -> (MatMut<'a, T>, MatMut<'a, T>) {
        assert!(
            i <= self.nrows,
            "Split index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        let top = MatMut {
            ptr: self.ptr,
            nrows: i,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        let bottom = MatMut {
            ptr: unsafe { self.ptr.offset(i as isize * self.row_stride) },
            nrows: self.nrows - i,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        (top, bottom)
    }

    pub fn split_at_col_mut(self, j: usize) -> (MatMut<'a, T>, MatMut<'a, T>) {
        assert!(
            j <= self.ncols,
            "Split index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        let left = MatMut {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: j,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        let right = MatMut {
            ptr: unsafe { self.ptr.offset(j as isize * self.col_stride) },
            nrows: self.nrows,
            ncols: self.ncols - j,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        };
        (left, right)
    }

    #[inline]
    pub fn transpose_mut(self) -> MatMut<'a, T> {
        MatMut {
            ptr: self.ptr,
            nrows: self.ncols,
            ncols: self.nrows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn reverse_rows_mut(self) -> MatMut<'a, T> {
        if self.nrows == 0 {
            return self;
        }
        MatMut {
            ptr: unsafe { self.ptr.offset((self.nrows - 1) as isize * self.row_stride) },
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: -self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn reverse_cols_mut(self) -> MatMut<'a, T> {
        if self.ncols == 0 {
            return self;
        }
        MatMut {
            ptr: unsafe { self.ptr.offset((self.ncols - 1) as isize * self.col_stride) },
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: -self.col_stride,
            _marker: PhantomData,
        }
    }

    pub fn copy_from(&mut self, src: MatRef<'_, T>)
    where
        T: Clone,
    {
        assert_eq!(
            (self.nrows, self.ncols),
            (src.nrows(), src.ncols()),
            "Shape mismatch: destination is {}x{} but source is {}x{}",
            self.nrows,
            self.ncols,
            src.nrows(),
            src.ncols()
        );
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                *self.at_mut(i, j) = src.at(i, j).clone();
            }
        }
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                *self.at_mut(i, j) = value.clone();
            }
        }
    }

    pub fn fill_with_fn<F: FnMut(usize, usize) -> T>(&mut self, mut f: F) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                *self.at_mut(i, j) = f(i, j);
            }
        }
    }

    pub fn swap_rows(&mut self, i1: usize, i2: usize) {
        assert!(
            i1 < self.nrows && i2 < self.nrows,
            "Row indices ({}, {}) out of bounds for {} rows",
            i1,
            i2,
            self.nrows
        );
        if i1 == i2 {
            return;
        }
        for j in 0..self.ncols {
            unsafe {
                std::ptr::swap(self.ptr_at_mut(i1, j), self.ptr_at_mut(i2, j));
            }
        }
    }

    pub fn swap_cols(&mut self, j1: usize, j2: usize) {
        assert!(
            j1 < self.ncols && j2 < self.ncols,
            "Column indices ({}, {}) out of bounds for {} columns",
            j1,
            j2,
            self.ncols
        );
        if j1 == j2 {
            return;
        }
        for i in 0..self.nrows {
            unsafe {
                std::ptr::swap(self.ptr_at_mut(i, j1), self.ptr_at_mut(i, j2));
            }
        }
    }
}

impl<T: PartialEq> MatMut<'_, T> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.rb().is_symmetric()
    }
}

impl<T: num_traits::Zero> MatMut<'_, T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.rb().is_diagonal()
    }

    #[inline]
    pub fn is_upper_triangular(&self) -> bool {
        self.rb().is_upper_triangular()
    }

    #[inline]
    pub fn is_lower_triangular(&self) -> bool {
        self.rb().is_lower_triangular()
    }
}

impl<T: PartialEq + num_traits::Zero + num_traits::One> MatMut<'_, T> {
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.rb().is_identity()
    }
}

impl<T: Clone + num_traits::Zero> MatMut<'_, T> {
    pub fn tril(&self, k: isize) -> Mat<T> {
        self.rb().tril(k)
    }

    pub fn triu(&self, k: isize) -> Mat<T> {
        self.rb().triu(k)
    }
}

impl<T: Clone> MatMut<'_, T> {
    pub fn take_rows(&self, indices: &[usize]) -> Mat<T> {
        self.rb().take_rows(indices)
    }

    pub fn take_cols(&self, indices: &[usize]) -> Mat<T> {
        self.rb().take_cols(indices)
    }

    pub fn reshape(&self, nrows: usize, ncols: usize) -> Mat<T> {
        self.rb().reshape(nrows, ncols)
    }

    pub fn flatten(&self) -> Mat<T> {
        self.rb().flatten()
    }

    pub fn flatten_row(&self) -> Mat<T> {
        self.rb().flatten_row()
    }

    pub fn to_col_vector(&self) -> Mat<T> {
        self.rb().to_col_vector()
    }

    pub fn to_row_vector(&self) -> Mat<T> {
        self.rb().to_row_vector()
    }

    pub fn insert_row(&self, i: usize, row: &[T]) -> Mat<T> {
        self.rb().insert_row(i, row)
    }

    pub fn insert_col(&self, j: usize, col: &[T]) -> Mat<T> {
        self.rb().insert_col(j, col)
    }

    pub fn remove_row(&self, i: usize) -> Mat<T> {
        self.rb().remove_row(i)
    }

    pub fn remove_col(&self, j: usize) -> Mat<T> {
        self.rb().remove_col(j)
    }

    pub fn append_row(&self, row: &[T]) -> Mat<T> {
        self.rb().append_row(row)
    }

    pub fn append_col(&self, col: &[T]) -> Mat<T> {
        self.rb().append_col(col)
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
