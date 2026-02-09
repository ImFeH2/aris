mod mat;
mod mat_mut;
mod mat_ref;

#[cfg(test)]
mod test;

use std::marker::PhantomData;

pub struct Mat<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
    col_stride: usize,
}

pub struct MatRef<'a, T> {
    ptr: *const T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    _marker: PhantomData<&'a T>,
}

impl<T> Copy for MatRef<'_, T> {}

impl<T> Clone for MatRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T: Sync> Send for MatRef<'_, T> {}
unsafe impl<T: Sync> Sync for MatRef<'_, T> {}

pub struct MatMut<'a, T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for MatMut<'_, T> {}
unsafe impl<T: Sync> Sync for MatMut<'_, T> {}

pub struct ColIter<'a, T> {
    matrix: MatRef<'a, T>,
    col: usize,
}

pub struct ColIterMut<'a, T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    col: usize,
    _marker: PhantomData<&'a mut T>,
}

pub struct RowIter<'a, T> {
    matrix: MatRef<'a, T>,
    row: usize,
}

pub struct RowIterMut<'a, T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    row: usize,
    _marker: PhantomData<&'a mut T>,
}

pub struct DiagIter<'a, T> {
    matrix: MatRef<'a, T>,
    index: usize,
    length: usize,
}

pub struct MatEnumerate<'a, T> {
    matrix: MatRef<'a, T>,
    index: usize,
    length: usize,
}

pub(crate) fn fmt_matrix<T: std::fmt::Display>(
    matrix: MatRef<'_, T>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    if matrix.nrows == 0 || matrix.ncols == 0 {
        return write!(f, "[]");
    }
    write!(f, "[")?;
    for i in 0..matrix.nrows {
        if i > 0 {
            write!(f, " ")?;
        }
        write!(f, "[")?;
        for j in 0..matrix.ncols {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", matrix.at(i, j))?;
        }
        write!(f, "]")?;
        if i < matrix.nrows - 1 {
            writeln!(f, ",")?;
        }
    }
    write!(f, "]")
}

pub(crate) fn fmt_matrix_debug<T: std::fmt::Debug>(
    type_name: &str,
    matrix: MatRef<'_, T>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    write!(f, "{}({}x{}, [", type_name, matrix.nrows, matrix.ncols)?;
    for i in 0..matrix.nrows {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "[")?;
        for j in 0..matrix.ncols {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", matrix.at(i, j))?;
        }
        write!(f, "]")?;
    }
    write!(f, "])")
}

#[macro_export]
macro_rules! mat {
    [$([$($elem:expr),* $(,)?]),+ $(,)?] => {{
        $crate::matrix::Mat::from_rows(&[$(&[$($elem),*][..]),+])
    }};

    ({$($block0:expr),+ $(,)?} $(, {$($block:expr),+ $(,)?})* $(,)?) => {{
        $crate::matrix::Mat::from_blocks(&[
            &[$(($block0).as_ref()),+][..],
            $(&[$(($block).as_ref()),+][..]),*
        ])
    }};

    [$elem:expr; $nrows:expr, $ncols:expr] => {{
        $crate::matrix::Mat::full($nrows, $ncols, $elem)
    }};

    ($nrows:expr, $ncols:expr, $f:expr) => {{
        $crate::matrix::Mat::from_fn($nrows, $ncols, $f)
    }};
}

#[macro_export]
macro_rules! col {
    [$($elem:expr),* $(,)?] => {{
        let data = vec![$($elem),*];
        let n = data.len();
        $crate::matrix::Mat::from_vec_col(n, 1, data)
    }};
}

#[macro_export]
macro_rules! row {
    [$($elem:expr),* $(,)?] => {{
        let data = vec![$($elem),*];
        let n = data.len();
        $crate::matrix::Mat::from_vec_row(1, n, data)
    }};
}
