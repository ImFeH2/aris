use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use num_traits::{One, Zero};

use super::{
    ColIter, ColIterMut, DiagIter, Mat, MatEnumerate, MatMut, MatRef, RowIter, RowIterMut,
    fmt_matrix, fmt_matrix_debug,
};

impl<T> Mat<T> {
    pub fn new() -> Self {
        Mat {
            data: Vec::new(),
            nrows: 0,
            ncols: 0,
            col_stride: 0,
        }
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
        1
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride as isize
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef {
            ptr: self.data.as_ptr(),
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: 1,
            col_stride: self.col_stride as isize,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, T> {
        MatMut {
            ptr: self.data.as_mut_ptr(),
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride: 1,
            col_stride: self.col_stride as isize,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.nrows && j < self.ncols {
            Some(&self.data[i + j * self.col_stride])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < self.nrows && j < self.ncols {
            Some(&mut self.data[i + j * self.col_stride])
        } else {
            None
        }
    }

    pub fn from_vec_col(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "Data length {} does not match {}x{}",
            data.len(),
            nrows,
            ncols
        );
        Mat {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        Mat {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }
}

impl<T: Clone> Mat<T> {
    pub fn full(nrows: usize, ncols: usize, value: T) -> Self {
        Mat {
            data: vec![value; nrows * ncols],
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_vec_row(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "Data length {} does not match {}x{}",
            data.len(),
            nrows,
            ncols
        );
        let mut col_major = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                col_major.push(data[i * ncols + j].clone());
            }
        }
        Mat {
            data: col_major,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_rows(rows: &[&[T]]) -> Self {
        let nrows = rows.len();
        if nrows == 0 {
            return Self::new();
        }
        let ncols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "Row {} has {} elements, expected {}",
                i,
                row.len(),
                ncols
            );
        }
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for row in rows {
                data.push(row[j].clone());
            }
        }
        Mat {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_cols(cols: &[&[T]]) -> Self {
        let ncols = cols.len();
        if ncols == 0 {
            return Self::new();
        }
        let nrows = cols[0].len();
        for (j, col) in cols.iter().enumerate() {
            assert_eq!(
                col.len(),
                nrows,
                "Column {} has {} elements, expected {}",
                j,
                col.len(),
                nrows
            );
        }
        let mut data = Vec::with_capacity(nrows * ncols);
        for col in cols {
            data.extend_from_slice(col);
        }
        Mat {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_nested_vec(rows: Vec<Vec<T>>) -> Self {
        let nrows = rows.len();
        if nrows == 0 {
            return Self::new();
        }
        let ncols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "Row {} has {} elements, expected {}",
                i,
                row.len(),
                ncols
            );
        }
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for row in &rows {
                data.push(row[j].clone());
            }
        }
        Mat {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn from_iter<I>(nrows: usize, ncols: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<T> = iter.into_iter().take(nrows * ncols).collect();
        assert_eq!(
            data.len(),
            nrows * ncols,
            "Iterator produced {} elements, expected {}",
            data.len(),
            nrows * ncols
        );
        Mat::from_vec_col(nrows, ncols, data)
    }

    pub fn from_blocks(block_rows: &[&[MatRef<'_, T>]]) -> Self {
        if block_rows.is_empty() {
            return Self::new();
        }

        let num_block_cols = block_rows[0].len();
        for (i, row) in block_rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                num_block_cols,
                "Block row {} has {} blocks, expected {}",
                i,
                row.len(),
                num_block_cols
            );
        }

        if num_block_cols == 0 {
            return Self::new();
        }

        let row_heights: Vec<usize> = block_rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let height = row[0].nrows();
                for (j, block) in row.iter().enumerate().skip(1) {
                    assert_eq!(
                        block.nrows(),
                        height,
                        "Block ({}, {}) has {} rows, expected {}",
                        i,
                        j,
                        block.nrows(),
                        height
                    );
                }
                height
            })
            .collect();

        let col_widths: Vec<usize> = (0..num_block_cols)
            .map(|j| {
                let width = block_rows[0][j].ncols();
                for (i, row) in block_rows.iter().enumerate().skip(1) {
                    assert_eq!(
                        row[j].ncols(),
                        width,
                        "Block ({}, {}) has {} columns, expected {}",
                        i,
                        j,
                        row[j].ncols(),
                        width
                    );
                }
                width
            })
            .collect();

        let total_rows: usize = row_heights.iter().sum();
        let total_cols: usize = col_widths.iter().sum();

        let mut data = Vec::with_capacity(total_rows * total_cols);
        for (bj, &col_width) in col_widths.iter().enumerate() {
            for local_j in 0..col_width {
                for (bi, &row_height) in row_heights.iter().enumerate() {
                    let block = block_rows[bi][bj];
                    for local_i in 0..row_height {
                        data.push(block.at(local_i, local_j).clone());
                    }
                }
            }
        }

        Mat::from_vec_col(total_rows, total_cols, data)
    }
}

impl<T: Clone> Mat<T> {
    pub fn copy_from(&mut self, src: MatRef<'_, T>) {
        self.as_mut().copy_from(src)
    }

    pub fn fill(&mut self, value: T) {
        self.as_mut().fill(value)
    }

    pub fn take_rows(&self, indices: &[usize]) -> Mat<T> {
        self.as_ref().take_rows(indices)
    }

    pub fn take_cols(&self, indices: &[usize]) -> Mat<T> {
        self.as_ref().take_cols(indices)
    }

    pub fn reshape(&self, nrows: usize, ncols: usize) -> Mat<T> {
        self.as_ref().reshape(nrows, ncols)
    }

    pub fn flatten(&self) -> Mat<T> {
        self.as_ref().flatten()
    }

    pub fn flatten_row(&self) -> Mat<T> {
        self.as_ref().flatten_row()
    }

    pub fn to_col_vector(&self) -> Mat<T> {
        self.as_ref().to_col_vector()
    }

    pub fn to_row_vector(&self) -> Mat<T> {
        self.as_ref().to_row_vector()
    }

    pub fn insert_row(&self, i: usize, row: &[T]) -> Mat<T> {
        self.as_ref().insert_row(i, row)
    }

    pub fn insert_col(&self, j: usize, col: &[T]) -> Mat<T> {
        self.as_ref().insert_col(j, col)
    }

    pub fn remove_row(&self, i: usize) -> Mat<T> {
        self.as_ref().remove_row(i)
    }

    pub fn remove_col(&self, j: usize) -> Mat<T> {
        self.as_ref().remove_col(j)
    }

    pub fn append_row(&self, row: &[T]) -> Mat<T> {
        self.as_ref().append_row(row)
    }

    pub fn append_col(&self, col: &[T]) -> Mat<T> {
        self.as_ref().append_col(col)
    }

    pub fn component_mul(&self, other: MatRef<'_, T>) -> Mat<T>
    where
        T: std::ops::Mul<Output = T>,
    {
        self.as_ref().component_mul(other)
    }

    pub fn component_div(&self, other: MatRef<'_, T>) -> Mat<T>
    where
        T: std::ops::Div<Output = T>,
    {
        self.as_ref().component_div(other)
    }

    pub fn clamp(&self, min: T, max: T) -> Mat<T>
    where
        T: PartialOrd,
    {
        self.as_ref().clamp(min, max)
    }

    pub fn vstack(matrices: &[MatRef<'_, T>]) -> Mat<T> {
        if matrices.is_empty() {
            return Mat::new();
        }
        let ncols = matrices[0].ncols();
        for (i, m) in matrices.iter().enumerate().skip(1) {
            assert_eq!(
                m.ncols(),
                ncols,
                "Matrix {} has {} columns, expected {}",
                i,
                m.ncols(),
                ncols
            );
        }
        let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
        let mut data = Vec::with_capacity(total_rows * ncols);
        for j in 0..ncols {
            for m in matrices {
                for i in 0..m.nrows() {
                    data.push(m.at(i, j).clone());
                }
            }
        }
        Mat::from_vec_col(total_rows, ncols, data)
    }

    pub fn hstack(matrices: &[MatRef<'_, T>]) -> Mat<T> {
        if matrices.is_empty() {
            return Mat::new();
        }
        let nrows = matrices[0].nrows();
        for (i, m) in matrices.iter().enumerate().skip(1) {
            assert_eq!(
                m.nrows(),
                nrows,
                "Matrix {} has {} rows, expected {}",
                i,
                m.nrows(),
                nrows
            );
        }
        let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
        let mut data = Vec::with_capacity(nrows * total_cols);
        for m in matrices {
            for j in 0..m.ncols() {
                for i in 0..nrows {
                    data.push(m.at(i, j).clone());
                }
            }
        }
        Mat::from_vec_col(nrows, total_cols, data)
    }

    pub fn resize(&mut self, nrows: usize, ncols: usize, fill_value: T) {
        let mut new_data = vec![fill_value; nrows * ncols];
        let copy_rows = self.nrows.min(nrows);
        let copy_cols = self.ncols.min(ncols);
        for j in 0..copy_cols {
            for i in 0..copy_rows {
                new_data[i + j * nrows] = self.data[i + j * self.col_stride].clone();
            }
        }
        self.data = new_data;
        self.nrows = nrows;
        self.ncols = ncols;
        self.col_stride = nrows;
    }

    pub fn truncate(&mut self, nrows: usize, ncols: usize) {
        assert!(
            nrows <= self.nrows && ncols <= self.ncols,
            "Cannot truncate {}x{} matrix to {}x{}",
            self.nrows,
            self.ncols,
            nrows,
            ncols
        );
        if nrows == self.nrows && ncols == self.ncols {
            return;
        }
        let mut new_data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                new_data.push(self.data[i + j * self.col_stride].clone());
            }
        }
        self.data = new_data;
        self.nrows = nrows;
        self.ncols = ncols;
        self.col_stride = nrows;
    }
}

impl<T> Mat<T> {
    pub fn reserve(&mut self, additional_cols: usize) {
        self.data.reserve(additional_cols * self.col_stride);
    }
}

impl<T: Clone + Zero> Mat<T> {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Mat {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    pub fn tril(&self, k: isize) -> Mat<T> {
        self.as_ref().tril(k)
    }

    pub fn triu(&self, k: isize) -> Mat<T> {
        self.as_ref().triu(k)
    }
}

impl<T: Clone + One> Mat<T> {
    pub fn ones(nrows: usize, ncols: usize) -> Self {
        Mat {
            data: vec![T::one(); nrows * ncols],
            nrows,
            ncols,
            col_stride: nrows,
        }
    }
}

impl<T: Clone + Zero + One> Mat<T> {
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i + i * n] = T::one();
        }
        m
    }

    pub fn eye(nrows: usize, ncols: usize, k: isize) -> Self {
        let mut m = Self::zeros(nrows, ncols);
        for i in 0..nrows {
            let j_signed = i as isize + k;
            if j_signed >= 0 {
                let j = j_signed as usize;
                if j < ncols {
                    m.data[i + j * nrows] = T::one();
                }
            }
        }
        m
    }

    pub fn diag(values: &[T]) -> Self {
        let n = values.len();
        let mut m = Self::zeros(n, n);
        for (k, v) in values.iter().enumerate() {
            m.data[k + k * n] = v.clone();
        }
        m
    }
}

impl<T> Mat<T> {
    pub fn col_iter(&self) -> ColIter<'_, T> {
        self.as_ref().col_iter()
    }

    pub fn row_iter(&self) -> RowIter<'_, T> {
        self.as_ref().row_iter()
    }

    pub fn diag_iter(&self) -> DiagIter<'_, T> {
        self.as_ref().diag_iter()
    }

    pub fn enumerate(&self) -> MatEnumerate<'_, T> {
        self.as_ref().enumerate()
    }

    pub fn col_iter_mut(&mut self) -> ColIterMut<'_, T> {
        self.as_mut().col_iter_mut()
    }

    pub fn row_iter_mut(&mut self) -> RowIterMut<'_, T> {
        self.as_mut().row_iter_mut()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        self.as_ref().is_square()
    }

    #[inline]
    pub fn is_row_vector(&self) -> bool {
        self.as_ref().is_row_vector()
    }

    #[inline]
    pub fn is_col_vector(&self) -> bool {
        self.as_ref().is_col_vector()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.as_ref().is_scalar()
    }

    #[inline]
    pub fn row(&self, i: usize) -> MatRef<'_, T> {
        self.as_ref().row(i)
    }

    #[inline]
    pub fn col(&self, j: usize) -> MatRef<'_, T> {
        self.as_ref().col(j)
    }

    #[inline]
    pub fn diagonal(&self) -> MatRef<'_, T> {
        self.as_ref().diagonal()
    }

    #[inline]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, T> {
        self.as_ref().submatrix(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn rows_range(&self, range: Range<usize>) -> MatRef<'_, T> {
        self.as_ref().rows_range(range)
    }

    #[inline]
    pub fn cols_range(&self, range: Range<usize>) -> MatRef<'_, T> {
        self.as_ref().cols_range(range)
    }

    #[inline]
    pub fn split_at_row(&self, i: usize) -> (MatRef<'_, T>, MatRef<'_, T>) {
        self.as_ref().split_at_row(i)
    }

    #[inline]
    pub fn split_at_col(&self, j: usize) -> (MatRef<'_, T>, MatRef<'_, T>) {
        self.as_ref().split_at_col(j)
    }

    #[inline]
    pub fn transpose(&self) -> MatRef<'_, T> {
        self.as_ref().transpose()
    }

    #[inline]
    pub fn reverse_rows(&self) -> MatRef<'_, T> {
        self.as_ref().reverse_rows()
    }

    #[inline]
    pub fn reverse_cols(&self) -> MatRef<'_, T> {
        self.as_ref().reverse_cols()
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> MatMut<'_, T> {
        self.as_mut().row_mut(i)
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> MatMut<'_, T> {
        self.as_mut().col_mut(j)
    }

    #[inline]
    pub fn diagonal_mut(&mut self) -> MatMut<'_, T> {
        self.as_mut().diagonal_mut()
    }

    #[inline]
    pub fn submatrix_mut(
        &mut self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'_, T> {
        self.as_mut()
            .submatrix_mut(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn rows_range_mut(&mut self, range: Range<usize>) -> MatMut<'_, T> {
        self.as_mut().rows_range_mut(range)
    }

    #[inline]
    pub fn cols_range_mut(&mut self, range: Range<usize>) -> MatMut<'_, T> {
        self.as_mut().cols_range_mut(range)
    }

    pub fn split_at_row_mut(&mut self, i: usize) -> (MatMut<'_, T>, MatMut<'_, T>) {
        self.as_mut().split_at_row_mut(i)
    }

    pub fn split_at_col_mut(&mut self, j: usize) -> (MatMut<'_, T>, MatMut<'_, T>) {
        self.as_mut().split_at_col_mut(j)
    }

    #[inline]
    pub fn transpose_mut(&mut self) -> MatMut<'_, T> {
        self.as_mut().transpose_mut()
    }

    #[inline]
    pub fn reverse_rows_mut(&mut self) -> MatMut<'_, T> {
        self.as_mut().reverse_rows_mut()
    }

    #[inline]
    pub fn reverse_cols_mut(&mut self) -> MatMut<'_, T> {
        self.as_mut().reverse_cols_mut()
    }

    pub fn fill_with_fn<F: FnMut(usize, usize) -> T>(&mut self, f: F) {
        self.as_mut().fill_with_fn(f)
    }

    pub fn swap_rows(&mut self, i1: usize, i2: usize) {
        self.as_mut().swap_rows(i1, i2)
    }

    pub fn swap_cols(&mut self, j1: usize, j2: usize) {
        self.as_mut().swap_cols(j1, j2)
    }

    pub fn map<U, F: FnMut(&T) -> U>(&self, f: F) -> Mat<U> {
        self.as_ref().map(f)
    }

    pub fn zip_map<U, F: FnMut(&T, &T) -> U>(&self, other: MatRef<'_, T>, f: F) -> Mat<U> {
        self.as_ref().zip_map(other, f)
    }
}

impl<T: num_traits::Signed + Clone> Mat<T> {
    pub fn abs(&self) -> Mat<T> {
        self.as_ref().abs()
    }

    pub fn signum(&self) -> Mat<T> {
        self.as_ref().signum()
    }
}

impl<T: num_traits::Float> Mat<T> {
    pub fn pow(&self, n: T) -> Mat<T> {
        self.as_ref().pow(n)
    }

    pub fn sqrt(&self) -> Mat<T> {
        self.as_ref().sqrt()
    }

    pub fn cbrt(&self) -> Mat<T> {
        self.as_ref().cbrt()
    }

    pub fn exp(&self) -> Mat<T> {
        self.as_ref().exp()
    }

    pub fn ln(&self) -> Mat<T> {
        self.as_ref().ln()
    }

    pub fn log10(&self) -> Mat<T> {
        self.as_ref().log10()
    }

    pub fn log2(&self) -> Mat<T> {
        self.as_ref().log2()
    }

    pub fn sin(&self) -> Mat<T> {
        self.as_ref().sin()
    }

    pub fn cos(&self) -> Mat<T> {
        self.as_ref().cos()
    }

    pub fn tan(&self) -> Mat<T> {
        self.as_ref().tan()
    }

    pub fn asin(&self) -> Mat<T> {
        self.as_ref().asin()
    }

    pub fn acos(&self) -> Mat<T> {
        self.as_ref().acos()
    }

    pub fn atan(&self) -> Mat<T> {
        self.as_ref().atan()
    }

    pub fn sinh(&self) -> Mat<T> {
        self.as_ref().sinh()
    }

    pub fn cosh(&self) -> Mat<T> {
        self.as_ref().cosh()
    }

    pub fn tanh(&self) -> Mat<T> {
        self.as_ref().tanh()
    }

    pub fn asinh(&self) -> Mat<T> {
        self.as_ref().asinh()
    }

    pub fn acosh(&self) -> Mat<T> {
        self.as_ref().acosh()
    }

    pub fn atanh(&self) -> Mat<T> {
        self.as_ref().atanh()
    }

    pub fn ceil(&self) -> Mat<T> {
        self.as_ref().ceil()
    }

    pub fn floor(&self) -> Mat<T> {
        self.as_ref().floor()
    }

    pub fn round(&self) -> Mat<T> {
        self.as_ref().round()
    }
}

impl<T: PartialEq> Mat<T> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.as_ref().is_symmetric()
    }
}

impl<T: Zero> Mat<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.as_ref().is_diagonal()
    }

    #[inline]
    pub fn is_upper_triangular(&self) -> bool {
        self.as_ref().is_upper_triangular()
    }

    #[inline]
    pub fn is_lower_triangular(&self) -> bool {
        self.as_ref().is_lower_triangular()
    }
}

impl<T: PartialEq + Zero + One> Mat<T> {
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.as_ref().is_identity()
    }
}

impl<T> Default for Mat<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for Mat<T> {
    fn clone(&self) -> Self {
        Mat {
            data: self.data.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }
}

impl<T: PartialEq> PartialEq for Mat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<T: Eq> Eq for Mat<T> {}

impl<T: PartialEq> PartialEq<MatRef<'_, T>> for Mat<T> {
    fn eq(&self, other: &MatRef<'_, T>) -> bool {
        self.as_ref() == *other
    }
}

impl<T: PartialEq> PartialEq<Mat<T>> for MatRef<'_, T> {
    fn eq(&self, other: &Mat<T>) -> bool {
        *self == other.as_ref()
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
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
        &self.data[i + j * self.col_stride]
    }
}

impl<T> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({}, {}) out of bounds for {}x{} matrix",
            i,
            j,
            self.nrows,
            self.ncols
        );
        &mut self.data[i + j * self.col_stride]
    }
}

impl<T: fmt::Display> fmt::Display for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(self.as_ref(), f)
    }
}

impl<T: fmt::Debug> fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix_debug("Mat", self.as_ref(), f)
    }
}

impl<T: Clone> From<MatRef<'_, T>> for Mat<T> {
    fn from(mat_ref: MatRef<'_, T>) -> Self {
        mat_ref.to_owned()
    }
}
