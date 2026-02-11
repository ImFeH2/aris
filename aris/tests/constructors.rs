use aris::{Mat, block, col, mat, row};

#[test]
fn block_macro_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5], [6]];
    let c = mat![[7, 8]];
    let d = mat![[9]];

    let m = block![[a, b], [c, d]];
    let expected = mat![[1, 2, 5], [3, 4, 6], [7, 8, 9]];
    assert_eq!(m, expected);
}

#[test]
fn block_macro_single_column() {
    let a = mat![[1, 2]];
    let b = mat![[3, 4]];
    let c = mat![[5, 6]];
    let m = block![[a], [b], [c]];
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn block_macro_single_row() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = block![[a, b]];
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8]];
    assert_eq!(m, expected);
}

#[test]
fn block_macro_trailing_comma() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = block![[a, b,], [b, a,],];
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8], [5, 6, 1, 2], [7, 8, 3, 4]];
    assert_eq!(m, expected);
}

#[test]
fn block_macro_with_identity() {
    let eye: Mat<i32> = Mat::identity(2);
    let z: Mat<i32> = Mat::zeros(2, 2);
    let m = block![[eye, z]];
    let expected = mat![[1, 0, 0, 0], [0, 1, 0, 0]];
    assert_eq!(m, expected);
}

#[test]
fn col_macro() {
    let c = col![1, 2, 3];
    assert_eq!(c.shape(), (3, 1));
    assert_eq!(c[(0, 0)], 1);
    assert_eq!(c[(1, 0)], 2);
    assert_eq!(c[(2, 0)], 3);
}

#[test]
fn col_major_storage_layout() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    assert_eq!(m.row_stride(), 1);
    assert_eq!(m.col_stride(), 2);
    assert_eq!(m.as_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn default_creates_empty_matrix() {
    let m: Mat<i32> = Mat::default();
    assert_eq!(m.shape(), (0, 0));
}

#[test]
fn diag_creates_diagonal_matrix() {
    let m: Mat<i32> = Mat::diag(&[2, 5, 8]);
    assert_eq!(m.shape(), (3, 3));
    assert_eq!(m[(0, 0)], 2);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(2, 2)], 8);
    assert_eq!(m[(0, 1)], 0);
    assert_eq!(m[(1, 0)], 0);
    assert_eq!(m[(0, 2)], 0);
}

#[test]
fn diag_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.diag_iter().count(), 0);
}

#[test]
fn diag_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.diag_iter();
    assert_eq!(iter.len(), 2);
}

#[test]
fn diag_iter_non_square() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&1, &5]);
}

#[test]
fn diag_iter_yields_diagonal() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&1, &5, &9]);
}

#[test]
fn eye_main_diagonal() {
    let m: Mat<i32> = Mat::eye(3, 4, 0);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 1)], 1);
    assert_eq!(m[(2, 2)], 1);
    assert_eq!(m[(0, 1)], 0);
    assert_eq!(m[(0, 3)], 0);
}

#[test]
fn eye_negative_offset() {
    let m: Mat<i32> = Mat::eye(4, 3, -1);
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(1, 0)], 1);
    assert_eq!(m[(2, 1)], 1);
    assert_eq!(m[(3, 2)], 1);
}

#[test]
fn eye_positive_offset() {
    let m: Mat<i32> = Mat::eye(3, 4, 1);
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(0, 1)], 1);
    assert_eq!(m[(1, 2)], 1);
    assert_eq!(m[(2, 3)], 1);
    assert_eq!(m[(1, 1)], 0);
}

#[test]
fn from_cols_creates_matrix() {
    let m = Mat::from_cols(&[&[1, 4], &[2, 5], &[3, 6]]);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(0, 2)], 3);
    assert_eq!(m[(1, 2)], 6);
}

#[test]
fn from_cols_empty() {
    let m: Mat<i32> = Mat::from_cols(&[]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
fn from_cols_equals_from_col_major() {
    let a = Mat::from_cols(&[&[1, 4], &[2, 5], &[3, 6]]);
    let b = Mat::from_vec_col(2, 3, vec![1, 4, 2, 5, 3, 6]);
    assert_eq!(a, b);
}

#[test]
#[should_panic(expected = "Column 1")]
fn from_cols_panics_on_inconsistent_lengths() {
    Mat::from_cols(&[&[1, 2], &[3]]);
}

#[test]
fn from_fn_generates_matrix() {
    let m = Mat::from_fn(3, 2, |i, j| (i * 10 + j) as f64);
    assert_eq!(m.shape(), (3, 2));
    assert_eq!(m[(0, 0)], 0.0);
    assert_eq!(m[(1, 0)], 10.0);
    assert_eq!(m[(2, 0)], 20.0);
    assert_eq!(m[(0, 1)], 1.0);
    assert_eq!(m[(1, 1)], 11.0);
    assert_eq!(m[(2, 1)], 21.0);
}

#[test]
fn from_mat_ref_to_mat() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let owned: Mat<i32> = Mat::from(r);
    assert_eq!(owned, m);
}

#[test]
fn from_rows_creates_matrix() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(0, 2)], 3);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(1, 2)], 6);
}

#[test]
fn from_rows_empty() {
    let m: Mat<i32> = Mat::from_rows(&[]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
fn from_rows_equals_from_row_major() {
    let a = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let b = Mat::from_vec_row(2, 3, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a, b);
}

#[test]
#[should_panic(expected = "Row 1")]
fn from_rows_panics_on_inconsistent_lengths() {
    Mat::from_rows(&[&[1, 2], &[3]]);
}

#[test]
#[should_panic(expected = "Data length")]
fn from_vec_col_panics_on_wrong_length() {
    Mat::from_vec_col(2, 3, vec![1, 2, 3]);
}

#[test]
fn from_vec_col_stores_correctly() {
    let m = Mat::from_vec_col(2, 3, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 0)], 2);
    assert_eq!(m[(0, 1)], 3);
    assert_eq!(m[(1, 1)], 4);
    assert_eq!(m[(0, 2)], 5);
    assert_eq!(m[(1, 2)], 6);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn from_vec_row_converts_to_vec_col() {
    let m = Mat::from_vec_row(2, 3, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(0, 2)], 3);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(1, 2)], 6);
    assert_eq!(m.as_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
#[should_panic(expected = "Data length")]
fn from_vec_row_panics_on_wrong_length() {
    Mat::from_vec_row(2, 3, vec![1, 2, 3, 4, 5]);
}

#[test]
fn full_creates_filled_matrix() {
    let m = Mat::full(2, 3, 42);
    assert_eq!(m.shape(), (2, 3));
    for j in 0..3 {
        for i in 0..2 {
            assert_eq!(m[(i, j)], 42);
        }
    }
}

#[test]
fn identity_1x1() {
    let m: Mat<i32> = Mat::identity(1);
    assert_eq!(m[(0, 0)], 1);
}

#[test]
fn identity_creates_identity_matrix() {
    let m: Mat<f64> = Mat::identity(3);
    assert_eq!(m.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(m[(i, j)], 1.0);
            } else {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }
}

#[test]
fn identity_diagonal_values() {
    let m: Mat<f64> = Mat::identity(4);
    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&1.0, &1.0, &1.0, &1.0]);

    for ((i, j), val) in m.enumerate() {
        if i == j {
            assert_eq!(*val, 1.0);
        } else {
            assert_eq!(*val, 0.0);
        }
    }
}

#[test]
fn mat_fill_macro() {
    let m = mat![7; 3, 2];
    assert_eq!(m.shape(), (3, 2));
    for j in 0..2 {
        for i in 0..3 {
            assert_eq!(m[(i, j)], 7);
        }
    }
}

#[test]
fn mat_fn_macro() {
    let m = mat!(2, 3, |i, j| i + j);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(0, 2)], 2);
    assert_eq!(m[(1, 1)], 2);
    assert_eq!(m[(1, 2)], 3);
}

#[test]
fn mat_literal_macro() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 2)], 6);
}

#[test]
fn mat_literal_macro_trailing_comma() {
    let m = mat![[1, 2], [3, 4],];
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m[(1, 1)], 4);
}

#[test]
fn new_creates_empty_matrix() {
    let m: Mat<f64> = Mat::new();
    assert_eq!(m.nrows(), 0);
    assert_eq!(m.ncols(), 0);
    assert_eq!(m.shape(), (0, 0));
    assert_eq!(m.size(), 0);
    assert!(m.as_slice().is_empty());
}

#[test]
fn ones_creates_ones_matrix() {
    let m: Mat<i32> = Mat::ones(2, 3);
    assert_eq!(m.shape(), (2, 3));
    for j in 0..3 {
        for i in 0..2 {
            assert_eq!(m[(i, j)], 1);
        }
    }
}

#[test]
fn row_macro() {
    let r = row![1, 2, 3];
    assert_eq!(r.shape(), (1, 3));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(0, 1)], 2);
    assert_eq!(r[(0, 2)], 3);
}

#[test]
fn zeros_0x0() {
    let m: Mat<f64> = Mat::zeros(0, 0);
    assert_eq!(m.shape(), (0, 0));
    assert_eq!(m.size(), 0);
}

#[test]
fn zeros_creates_zero_matrix() {
    let m: Mat<f64> = Mat::zeros(3, 4);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m.size(), 12);
    for j in 0..4 {
        for i in 0..3 {
            assert_eq!(m[(i, j)], 0.0);
        }
    }
}

#[test]
fn from_nested_vec_2x3() {
    let rows = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let m = Mat::from_nested_vec(rows);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(0, 2)], 3);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(1, 2)], 6);
}

#[test]
fn from_nested_vec_empty() {
    let rows: Vec<Vec<i32>> = vec![];
    let m = Mat::from_nested_vec(rows);
    assert_eq!(m.shape(), (0, 0));
    assert_eq!(m.size(), 0);
}

#[test]
fn from_nested_vec_1x1() {
    let rows = vec![vec![42]];
    let m = Mat::from_nested_vec(rows);
    assert_eq!(m.shape(), (1, 1));
    assert_eq!(m[(0, 0)], 42);
}

#[test]
fn from_nested_vec_3x2() {
    let rows = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let m = Mat::from_nested_vec(rows);
    assert_eq!(m.shape(), (3, 2));
    assert_eq!(m, mat![[1, 2], [3, 4], [5, 6]]);
}

#[test]
#[should_panic(expected = "Row 1 has 2 elements, expected 3")]
fn from_nested_vec_inconsistent_row_length() {
    let rows = vec![vec![1, 2, 3], vec![4, 5]];
    Mat::from_nested_vec(rows);
}

#[test]
fn from_iter_2x3() {
    let m = Mat::from_iter(2, 3, 0..6);
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m, mat![[0, 2, 4], [1, 3, 5]]);
}

#[test]
fn from_iter_3x3() {
    let m = Mat::from_iter(3, 3, 1..10);
    assert_eq!(m.shape(), (3, 3));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 0)], 2);
    assert_eq!(m[(2, 0)], 3);
    assert_eq!(m[(0, 1)], 4);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(2, 1)], 6);
    assert_eq!(m[(0, 2)], 7);
    assert_eq!(m[(1, 2)], 8);
    assert_eq!(m[(2, 2)], 9);
}

#[test]
fn from_iter_1x5() {
    let m = Mat::from_iter(1, 5, 10..15);
    assert_eq!(m.shape(), (1, 5));
    assert_eq!(m, mat![[10, 11, 12, 13, 14]]);
}

#[test]
#[should_panic(expected = "Iterator produced 5 elements, expected 6")]
fn from_iter_insufficient_elements() {
    Mat::from_iter(2, 3, 0..5);
}

#[test]
fn from_iter_with_iterator() {
    let v = vec![1, 2, 3, 4];
    let m = Mat::from_iter(2, 2, v);
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m, mat![[1, 3], [2, 4]]);
}
