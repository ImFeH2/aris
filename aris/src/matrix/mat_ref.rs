use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, Range};

use num_traits::{One, Zero};

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

    #[inline]
    pub fn is_empty(self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    #[inline]
    pub fn is_square(self) -> bool {
        self.nrows == self.ncols
    }

    #[inline]
    pub fn is_row_vector(self) -> bool {
        self.nrows == 1
    }

    #[inline]
    pub fn is_col_vector(self) -> bool {
        self.ncols == 1
    }

    #[inline]
    pub fn is_scalar(self) -> bool {
        self.nrows == 1 && self.ncols == 1
    }

    #[inline]
    pub fn row(self, i: usize) -> MatRef<'a, T> {
        assert!(
            i < self.nrows,
            "Row index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        MatRef {
            ptr: self.ptr_at(i, 0),
            nrows: 1,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn col(self, j: usize) -> MatRef<'a, T> {
        assert!(
            j < self.ncols,
            "Column index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        MatRef {
            ptr: self.ptr_at(0, j),
            nrows: self.nrows,
            ncols: 1,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn diagonal(self) -> MatRef<'a, T> {
        let len = self.nrows.min(self.ncols);
        MatRef {
            ptr: self.ptr,
            nrows: len,
            ncols: 1,
            row_stride: self.row_stride + self.col_stride,
            col_stride: self.row_stride + self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn submatrix(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'a, T> {
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
        MatRef {
            ptr: self.ptr_at(row_start, col_start),
            nrows,
            ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn rows_range(self, range: Range<usize>) -> MatRef<'a, T> {
        assert!(
            range.start <= range.end,
            "Invalid range: start {} > end {}",
            range.start,
            range.end
        );
        self.submatrix(range.start, 0, range.end - range.start, self.ncols)
    }

    #[inline]
    pub fn cols_range(self, range: Range<usize>) -> MatRef<'a, T> {
        assert!(
            range.start <= range.end,
            "Invalid range: start {} > end {}",
            range.start,
            range.end
        );
        self.submatrix(0, range.start, self.nrows, range.end - range.start)
    }

    #[inline]
    pub fn split_at_row(self, i: usize) -> (MatRef<'a, T>, MatRef<'a, T>) {
        assert!(
            i <= self.nrows,
            "Split index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        (
            self.submatrix(0, 0, i, self.ncols),
            self.submatrix(i, 0, self.nrows - i, self.ncols),
        )
    }

    #[inline]
    pub fn split_at_col(self, j: usize) -> (MatRef<'a, T>, MatRef<'a, T>) {
        assert!(
            j <= self.ncols,
            "Split index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        (
            self.submatrix(0, 0, self.nrows, j),
            self.submatrix(0, j, self.nrows, self.ncols - j),
        )
    }

    #[inline]
    pub fn transpose(self) -> MatRef<'a, T> {
        MatRef {
            ptr: self.ptr,
            nrows: self.ncols,
            ncols: self.nrows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn reverse_rows(self) -> MatRef<'a, T> {
        if self.nrows == 0 {
            return self;
        }
        MatRef {
            ptr: self.ptr_at(self.nrows - 1, 0),
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: -self.row_stride,
            col_stride: self.col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn reverse_cols(self) -> MatRef<'a, T> {
        if self.ncols == 0 {
            return self;
        }
        MatRef {
            ptr: self.ptr_at(0, self.ncols - 1),
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: self.row_stride,
            col_stride: -self.col_stride,
            _marker: PhantomData,
        }
    }

    pub fn map<U, F: FnMut(&T) -> U>(self, mut f: F) -> Mat<U> {
        let (nrows, ncols) = self.shape();
        let mut data = Vec::with_capacity(self.size());
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(self.at(i, j)));
            }
        }
        Mat::from_vec_col(nrows, ncols, data)
    }

    pub fn zip_map<U, F: FnMut(&T, &T) -> U>(self, other: MatRef<'_, T>, mut f: F) -> Mat<U> {
        assert_eq!(
            self.shape(),
            other.shape(),
            "shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        let (nrows, ncols) = self.shape();
        let mut data = Vec::with_capacity(self.size());
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(self.at(i, j), other.at(i, j)));
            }
        }
        Mat::from_vec_col(nrows, ncols, data)
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

impl<T: PartialEq> MatRef<'_, T> {
    pub fn is_symmetric(self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        for j in 1..self.ncols {
            for i in 0..j {
                if self.at(i, j) != self.at(j, i) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Zero> MatRef<'_, T> {
    pub fn is_diagonal(self) -> bool {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                if i != j && !self.at(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_upper_triangular(self) -> bool {
        for j in 0..self.ncols {
            for i in (j + 1)..self.nrows {
                if !self.at(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_lower_triangular(self) -> bool {
        for j in 0..self.ncols {
            for i in 0..j.min(self.nrows) {
                if !self.at(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: PartialEq + Zero + One> MatRef<'_, T> {
    pub fn is_identity(self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        let one = T::one();
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                if i == j {
                    if *self.at(i, j) != one {
                        return false;
                    }
                } else if !self.at(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T: Clone + Zero> MatRef<'a, T> {
    pub fn tril(self, k: isize) -> Mat<T> {
        Mat::from_fn(self.nrows, self.ncols, |i, j| {
            if (j as isize) - (i as isize) <= k {
                self.at(i, j).clone()
            } else {
                T::zero()
            }
        })
    }

    pub fn triu(self, k: isize) -> Mat<T> {
        Mat::from_fn(self.nrows, self.ncols, |i, j| {
            if (j as isize) - (i as isize) >= k {
                self.at(i, j).clone()
            } else {
                T::zero()
            }
        })
    }
}

impl<'a, T: Clone> MatRef<'a, T> {
    pub fn take_rows(self, indices: &[usize]) -> Mat<T> {
        for &i in indices {
            assert!(
                i < self.nrows,
                "Row index {} out of bounds for {} rows",
                i,
                self.nrows
            );
        }
        let nrows = indices.len();
        let mut data = Vec::with_capacity(nrows * self.ncols);
        for j in 0..self.ncols {
            for &i in indices {
                data.push(self.at(i, j).clone());
            }
        }
        Mat::from_vec_col(nrows, self.ncols, data)
    }

    pub fn take_cols(self, indices: &[usize]) -> Mat<T> {
        for &j in indices {
            assert!(
                j < self.ncols,
                "Column index {} out of bounds for {} columns",
                j,
                self.ncols
            );
        }
        let ncols = indices.len();
        let mut data = Vec::with_capacity(self.nrows * ncols);
        for &j in indices {
            for i in 0..self.nrows {
                data.push(self.at(i, j).clone());
            }
        }
        Mat::from_vec_col(self.nrows, ncols, data)
    }

    pub fn reshape(self, nrows: usize, ncols: usize) -> Mat<T> {
        assert_eq!(
            self.nrows * self.ncols,
            nrows * ncols,
            "Cannot reshape {}x{} ({} elements) to {}x{} ({} elements)",
            self.nrows,
            self.ncols,
            self.nrows * self.ncols,
            nrows,
            ncols,
            nrows * ncols
        );
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(self.at(i, j).clone());
            }
        }
        Mat::from_vec_col(nrows, ncols, data)
    }

    pub fn flatten(self) -> Mat<T> {
        let size = self.nrows * self.ncols;
        self.reshape(size, 1)
    }

    pub fn flatten_row(self) -> Mat<T> {
        let size = self.nrows * self.ncols;
        self.reshape(1, size)
    }

    pub fn to_col_vector(self) -> Mat<T> {
        self.flatten()
    }

    pub fn to_row_vector(self) -> Mat<T> {
        self.flatten_row()
    }

    pub fn insert_row(self, i: usize, row: &[T]) -> Mat<T> {
        assert!(
            i <= self.nrows,
            "Row insert index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        assert_eq!(
            row.len(),
            self.ncols,
            "Row length {} does not match {} columns",
            row.len(),
            self.ncols
        );
        let new_nrows = self.nrows + 1;
        let mut data = Vec::with_capacity(new_nrows * self.ncols);
        for (j, val) in row.iter().enumerate() {
            for r in 0..i {
                data.push(self.at(r, j).clone());
            }
            data.push(val.clone());
            for r in i..self.nrows {
                data.push(self.at(r, j).clone());
            }
        }
        Mat::from_vec_col(new_nrows, self.ncols, data)
    }

    pub fn insert_col(self, j: usize, col: &[T]) -> Mat<T> {
        assert!(
            j <= self.ncols,
            "Column insert index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        assert_eq!(
            col.len(),
            self.nrows,
            "Column length {} does not match {} rows",
            col.len(),
            self.nrows
        );
        let new_ncols = self.ncols + 1;
        let mut data = Vec::with_capacity(self.nrows * new_ncols);
        for c in 0..j {
            for r in 0..self.nrows {
                data.push(self.at(r, c).clone());
            }
        }
        for val in col.iter().take(self.nrows) {
            data.push(val.clone());
        }
        for c in j..self.ncols {
            for r in 0..self.nrows {
                data.push(self.at(r, c).clone());
            }
        }
        Mat::from_vec_col(self.nrows, new_ncols, data)
    }

    pub fn remove_row(self, i: usize) -> Mat<T> {
        assert!(
            i < self.nrows,
            "Row index {} out of bounds for {} rows",
            i,
            self.nrows
        );
        let new_nrows = self.nrows - 1;
        let mut data = Vec::with_capacity(new_nrows * self.ncols);
        for j in 0..self.ncols {
            for r in 0..self.nrows {
                if r != i {
                    data.push(self.at(r, j).clone());
                }
            }
        }
        Mat::from_vec_col(new_nrows, self.ncols, data)
    }

    pub fn remove_col(self, j: usize) -> Mat<T> {
        assert!(
            j < self.ncols,
            "Column index {} out of bounds for {} columns",
            j,
            self.ncols
        );
        let new_ncols = self.ncols - 1;
        let mut data = Vec::with_capacity(self.nrows * new_ncols);
        for c in 0..self.ncols {
            if c != j {
                for r in 0..self.nrows {
                    data.push(self.at(r, c).clone());
                }
            }
        }
        Mat::from_vec_col(self.nrows, new_ncols, data)
    }

    pub fn append_row(self, row: &[T]) -> Mat<T> {
        let nrows = self.nrows;
        self.insert_row(nrows, row)
    }

    pub fn append_col(self, col: &[T]) -> Mat<T> {
        let ncols = self.ncols;
        self.insert_col(ncols, col)
    }

    pub fn element_mul(self, other: MatRef<'_, T>) -> Mat<T>
    where
        T: std::ops::Mul<Output = T>,
    {
        self.zip_map(other, |a, b| a.clone() * b.clone())
    }

    pub fn element_div(self, other: MatRef<'_, T>) -> Mat<T>
    where
        T: std::ops::Div<Output = T>,
    {
        self.zip_map(other, |a, b| a.clone() / b.clone())
    }

    pub fn clamp(self, min: T, max: T) -> Mat<T>
    where
        T: PartialOrd,
    {
        self.map(|x| {
            if *x < min {
                min.clone()
            } else if *x > max {
                max.clone()
            } else {
                x.clone()
            }
        })
    }
}

impl<'a, T: num_traits::Signed + Clone> MatRef<'a, T> {
    pub fn abs(self) -> Mat<T> {
        self.map(|x| x.abs())
    }

    pub fn signum(self) -> Mat<T> {
        self.map(|x| x.signum())
    }
}

impl<'a, T: num_traits::Float> MatRef<'a, T> {
    pub fn pow(self, n: T) -> Mat<T> {
        self.map(|x| x.powf(n))
    }

    pub fn sqrt(self) -> Mat<T> {
        self.map(|x| x.sqrt())
    }

    pub fn cbrt(self) -> Mat<T> {
        self.map(|x| x.cbrt())
    }

    pub fn exp(self) -> Mat<T> {
        self.map(|x| x.exp())
    }

    pub fn ln(self) -> Mat<T> {
        self.map(|x| x.ln())
    }

    pub fn log10(self) -> Mat<T> {
        self.map(|x| x.log10())
    }

    pub fn log2(self) -> Mat<T> {
        self.map(|x| x.log2())
    }

    pub fn sin(self) -> Mat<T> {
        self.map(|x| x.sin())
    }

    pub fn cos(self) -> Mat<T> {
        self.map(|x| x.cos())
    }

    pub fn tan(self) -> Mat<T> {
        self.map(|x| x.tan())
    }

    pub fn asin(self) -> Mat<T> {
        self.map(|x| x.asin())
    }

    pub fn acos(self) -> Mat<T> {
        self.map(|x| x.acos())
    }

    pub fn atan(self) -> Mat<T> {
        self.map(|x| x.atan())
    }

    pub fn sinh(self) -> Mat<T> {
        self.map(|x| x.sinh())
    }

    pub fn cosh(self) -> Mat<T> {
        self.map(|x| x.cosh())
    }

    pub fn tanh(self) -> Mat<T> {
        self.map(|x| x.tanh())
    }

    pub fn asinh(self) -> Mat<T> {
        self.map(|x| x.asinh())
    }

    pub fn acosh(self) -> Mat<T> {
        self.map(|x| x.acosh())
    }

    pub fn atanh(self) -> Mat<T> {
        self.map(|x| x.atanh())
    }

    pub fn ceil(self) -> Mat<T> {
        self.map(|x| x.ceil())
    }

    pub fn floor(self) -> Mat<T> {
        self.map(|x| x.floor())
    }

    pub fn round(self) -> Mat<T> {
        self.map(|x| x.round())
    }
}

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
