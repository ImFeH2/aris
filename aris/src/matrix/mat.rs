use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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

impl<T: Clone + Zero> Mat<T> {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Mat {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
            col_stride: nrows,
        }
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
