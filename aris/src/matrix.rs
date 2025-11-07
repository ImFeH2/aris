use num_traits::{Num, real::Real};
use std::ops::{Add, Index, IndexMut, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>
where
    T: Num + Clone,
{
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Num + Clone,
{
    pub fn zeros((rows, cols): (usize, usize)) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }
}

impl<T> Matrix<T>
where
    T: Num + Clone,
{
    pub fn from_vec((rows, cols): (usize, usize), data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix { rows, cols, data }
    }

    pub fn from_nested_vec(nested_data: Vec<Vec<T>>) -> Self {
        let rows = nested_data.len();
        let cols = if rows > 0 { nested_data[0].len() } else { 0 };

        #[cfg(debug_assertions)]
        for (i, row) in nested_data.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "Row {} has {} elements, expected {}",
                i,
                row.len(),
                cols
            );
        }

        let data: Vec<T> = nested_data.into_iter().flatten().collect();
        Matrix { rows, cols, data }
    }

    pub fn from_value((rows, cols): (usize, usize), value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn from_fn<F>((rows, cols): (usize, usize), mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data: Vec<T> = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
    }

    pub fn get(&self, (row, col): (usize, usize)) -> Option<&T> {
        self.data.get(row * self.cols + col)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn transpose(&self) -> Matrix<T> {
        let mut data: Vec<T> = Vec::with_capacity(self.rows * self.cols);

        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self[(i, j)].clone());
            }
        }

        Matrix::from_vec((self.cols, self.rows), data)
    }

    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows);

        let mut data: Vec<T> = Vec::with_capacity(self.rows * other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    sum = sum + self[(i, k)].clone() * other[(k, j)].clone();
                }
                data.push(sum);
            }
        }

        Matrix::from_vec((self.rows, other.cols), data)
    }

    pub fn component_mul(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();

        Matrix::from_vec((self.rows, self.cols), data)
    }
}

impl<T> Matrix<T>
where
    T: Real,
{
    pub fn norm(&self) -> T {
        let mut sum = T::zero();
        for value in &self.data {
            sum = sum + *value * *value;
        }
        sum.sqrt()
    }
}

impl<T> std::fmt::Display for Matrix<T>
where
    T: Num + Clone + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Num + Clone,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Num + Clone,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl<T> Add<&Matrix<T>> for &Matrix<T>
where
    T: Num + Clone,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let data: Vec<T> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix::from_vec((self.rows, self.cols), data)
    }
}

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Num + Clone,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let data: Vec<T> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Matrix::from_vec((self.rows, self.cols), data)
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Num + Clone,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        self.dot(rhs)
    }
}

macro_rules! derive_binary_op {
    ($trait_name:ident, $method:ident, $operator:tt) => {
        impl<T> $trait_name<Matrix<T>> for Matrix<T>
        where
            T: Num + Clone,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                &self $operator &rhs
            }
        }

        impl<T> $trait_name<&Matrix<T>> for Matrix<T>
        where
            T: Num + Clone,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T> $trait_name<Matrix<T>> for &Matrix<T>
        where
            T: Num + Clone,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                self $operator &rhs
            }
        }
    };
}

derive_binary_op!(Add, add, +);
derive_binary_op!(Sub, sub, -);
derive_binary_op!(Mul, mul, *);

#[macro_export]
macro_rules! mat {
    [$elem:expr; $rows:expr, $cols:expr] => {{
        Matrix::from_value(($rows, $cols), $elem)
    }};

    [$([$($elem:expr),* $(,)?]),+ $(,)?] => {{
        let nested_vec = vec![$(vec![$($elem),*]),+];
        Matrix::from_nested_vec(nested_vec)
    }};
}
