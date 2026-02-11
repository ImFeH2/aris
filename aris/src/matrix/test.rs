use super::*;
use crate::{block, col, mat, row};
use num_complex::Complex;

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
fn default_creates_empty_matrix() {
    let m: Mat<i32> = Mat::default();
    assert_eq!(m.shape(), (0, 0));
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
fn identity_1x1() {
    let m: Mat<i32> = Mat::identity(1);
    assert_eq!(m[(0, 0)], 1);
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
fn eye_positive_offset() {
    let m: Mat<i32> = Mat::eye(3, 4, 1);
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(0, 1)], 1);
    assert_eq!(m[(1, 2)], 1);
    assert_eq!(m[(2, 3)], 1);
    assert_eq!(m[(1, 1)], 0);
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
#[should_panic(expected = "Data length")]
fn from_vec_col_panics_on_wrong_length() {
    Mat::from_vec_col(2, 3, vec![1, 2, 3]);
}

#[test]
#[should_panic(expected = "Data length")]
fn from_vec_row_panics_on_wrong_length() {
    Mat::from_vec_row(2, 3, vec![1, 2, 3, 4, 5]);
}

#[test]
#[should_panic(expected = "Row 1")]
fn from_rows_panics_on_inconsistent_lengths() {
    Mat::from_rows(&[&[1, 2], &[3]]);
}

#[test]
#[should_panic(expected = "Column 1")]
fn from_cols_panics_on_inconsistent_lengths() {
    Mat::from_cols(&[&[1, 2], &[3]]);
}

#[test]
fn col_major_storage_layout() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    assert_eq!(m.row_stride(), 1);
    assert_eq!(m.col_stride(), 2);
    assert_eq!(m.as_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn index_and_index_mut() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 1)], 4);
    m[(0, 1)] = 20;
    assert_eq!(m[(0, 1)], 20);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn index_out_of_bounds_row() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let _ = m[(2, 0)];
}

#[test]
#[should_panic(expected = "out of bounds")]
fn index_out_of_bounds_col() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let _ = m[(0, 2)];
}

#[test]
fn get_returns_some_for_valid_indices() {
    let m = Mat::from_rows(&[&[10, 20], &[30, 40]]);
    assert_eq!(m.get(0, 0), Some(&10));
    assert_eq!(m.get(1, 1), Some(&40));
}

#[test]
fn get_returns_none_for_invalid_indices() {
    let m = Mat::from_rows(&[&[10, 20], &[30, 40]]);
    assert_eq!(m.get(2, 0), None);
    assert_eq!(m.get(0, 2), None);
}

#[test]
fn get_mut_modifies_element() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    *m.get_mut(1, 0).unwrap() = 99;
    assert_eq!(m[(1, 0)], 99);
}

#[test]
fn as_slice_mut_modifies_data() {
    let mut m = Mat::zeros(2, 2);
    m.as_slice_mut()[0] = 7.0;
    assert_eq!(m[(0, 0)], 7.0);
}

#[test]
fn as_ref_creates_view() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 2);
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 1)], 4);
}

#[test]
fn as_mut_creates_mutable_view() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    {
        let mut v = m.as_mut();
        v[(0, 1)] = 20;
    }
    assert_eq!(m[(0, 1)], 20);
}

#[test]
fn mat_ref_get() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let r = m.as_ref();
    assert_eq!(r.get(0, 2), Some(&3));
    assert_eq!(r.get(2, 0), None);
}

#[test]
fn mat_mut_get_and_get_mut() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    {
        let mut v = m.as_mut();
        assert_eq!(v.get(1, 0), Some(&3));
        assert_eq!(v.get(2, 0), None);
        *v.get_mut(1, 1).unwrap() = 44;
    }
    assert_eq!(m[(1, 1)], 44);
}

#[test]
fn mat_mut_rb_creates_ref_view() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    let r = v.rb();
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r.shape(), (2, 2));
}

#[test]
fn mat_mut_rb_mut_creates_reborrow() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    {
        let mut v = m.as_mut();
        {
            let mut rb = v.rb_mut();
            rb[(0, 0)] = 10;
        }
    }
    assert_eq!(m[(0, 0)], 10);
}

#[test]
fn to_owned_creates_independent_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let owned = r.to_owned();
    assert_eq!(owned, m);
    assert_eq!(owned.shape(), m.shape());
}

#[test]
fn col_iter_yields_columns() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let cols: Vec<_> = m.col_iter().collect();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0][(0, 0)], 1);
    assert_eq!(cols[0][(1, 0)], 4);
    assert_eq!(cols[1][(0, 0)], 2);
    assert_eq!(cols[1][(1, 0)], 5);
    assert_eq!(cols[2][(0, 0)], 3);
    assert_eq!(cols[2][(1, 0)], 6);
}

#[test]
fn col_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.col_iter();
    assert_eq!(iter.len(), 3);
}

#[test]
fn row_iter_yields_rows() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let rows: Vec<_> = m.row_iter().collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][(0, 0)], 1);
    assert_eq!(rows[0][(0, 1)], 2);
    assert_eq!(rows[0][(0, 2)], 3);
    assert_eq!(rows[1][(0, 0)], 4);
    assert_eq!(rows[1][(0, 1)], 5);
    assert_eq!(rows[1][(0, 2)], 6);
}

#[test]
fn row_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.row_iter();
    assert_eq!(iter.len(), 2);
}

#[test]
fn diag_iter_yields_diagonal() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&1, &5, &9]);
}

#[test]
fn diag_iter_non_square() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&1, &5]);
}

#[test]
fn diag_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.diag_iter();
    assert_eq!(iter.len(), 2);
}

#[test]
fn enumerate_yields_row_major_order() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let items: Vec<_> = m.enumerate().collect();
    assert_eq!(
        items,
        vec![((0, 0), &1), ((0, 1), &2), ((1, 0), &3), ((1, 1), &4),]
    );
}

#[test]
fn enumerate_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.enumerate();
    assert_eq!(iter.len(), 6);
}

#[test]
fn col_iter_mut_modifies_columns() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    for mut col in m.col_iter_mut() {
        col[(0, 0)] *= 10;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(1, 0)], 3);
    assert_eq!(m[(1, 1)], 4);
}

#[test]
fn row_iter_mut_modifies_rows() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    for mut row in m.row_iter_mut() {
        row[(0, 0)] *= 10;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(1, 0)], 30);
    assert_eq!(m[(1, 1)], 4);
}

#[test]
fn clone_creates_independent_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let c = m.clone();
    assert_eq!(m, c);
}

#[test]
fn partial_eq_same_content() {
    let a = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let b = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(a, b);
}

#[test]
fn partial_eq_different_content() {
    let a = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let b = Mat::from_rows(&[&[1, 2], &[3, 5]]);
    assert_ne!(a, b);
}

#[test]
fn partial_eq_different_shape() {
    let a = Mat::from_rows(&[&[1, 2, 3]]);
    let b = Mat::from_rows(&[&[1], &[2], &[3]]);
    assert_ne!(a, b);
}

#[test]
fn partial_eq_mat_and_mat_ref() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(m, r);
    assert_eq!(r, m);
}

#[test]
fn mat_ref_partial_eq() {
    let a = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let b = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(a.as_ref(), b.as_ref());
}

#[test]
fn from_mat_ref_to_mat() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let owned: Mat<i32> = Mat::from(r);
    assert_eq!(owned, m);
}

#[test]
fn mat_ref_is_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let r2 = r;
    assert_eq!(r[(0, 0)], r2[(0, 0)]);
}

#[test]
fn display_format() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let s = format!("{}", m);
    assert_eq!(s, "[[1, 2],\n [3, 4]]");
}

#[test]
fn display_single_row() {
    let m = Mat::from_rows(&[&[1, 2, 3]]);
    let s = format!("{}", m);
    assert_eq!(s, "[[1, 2, 3]]");
}

#[test]
fn display_empty() {
    let m: Mat<i32> = Mat::new();
    let s = format!("{}", m);
    assert_eq!(s, "[]");
}

#[test]
fn debug_format() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let s = format!("{:?}", m);
    assert_eq!(s, "Mat(2x2, [[1, 2], [3, 4]])");
}

#[test]
fn mat_ref_display() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(format!("{}", r), "[[1, 2],\n [3, 4]]");
}

#[test]
fn mat_ref_debug() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(format!("{:?}", r), "MatRef(2x2, [[1, 2], [3, 4]])");
}

#[test]
fn mat_mut_display() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    assert_eq!(format!("{}", v), "[[1, 2],\n [3, 4]]");
}

#[test]
fn mat_mut_debug() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    assert_eq!(format!("{:?}", v), "MatMut(2x2, [[1, 2], [3, 4]])");
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
fn col_macro() {
    let c = col![1, 2, 3];
    assert_eq!(c.shape(), (3, 1));
    assert_eq!(c[(0, 0)], 1);
    assert_eq!(c[(1, 0)], 2);
    assert_eq!(c[(2, 0)], 3);
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
fn single_element_matrix() {
    let m = mat![[42]];
    assert_eq!(m.shape(), (1, 1));
    assert_eq!(m[(0, 0)], 42);
    assert_eq!(m.as_slice(), &[42]);

    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&42]);
}

#[test]
fn zeros_0x0() {
    let m: Mat<f64> = Mat::zeros(0, 0);
    assert_eq!(m.shape(), (0, 0));
    assert_eq!(m.size(), 0);
}

#[test]
fn col_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.col_iter().count(), 0);
}

#[test]
fn row_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.row_iter().count(), 0);
}

#[test]
fn diag_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.diag_iter().count(), 0);
}

#[test]
fn enumerate_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.enumerate().count(), 0);
}

#[test]
fn col_vector_operations() {
    let c = col![10, 20, 30];
    assert_eq!(c.nrows(), 3);
    assert_eq!(c.ncols(), 1);
    let cols: Vec<_> = c.col_iter().collect();
    assert_eq!(cols.len(), 1);
    assert_eq!(cols[0][(0, 0)], 10);
    assert_eq!(cols[0][(1, 0)], 20);
    assert_eq!(cols[0][(2, 0)], 30);
}

#[test]
fn row_vector_operations() {
    let r = row![10, 20, 30];
    assert_eq!(r.nrows(), 1);
    assert_eq!(r.ncols(), 3);
    let rows: Vec<_> = r.row_iter().collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][(0, 0)], 10);
    assert_eq!(rows[0][(0, 1)], 20);
    assert_eq!(rows[0][(0, 2)], 30);
}

#[test]
fn mat_ref_index_panics() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let result = std::panic::catch_unwind(|| r[(2, 0)]);
    assert!(result.is_err());
}

#[test]
fn mat_mut_index_panics() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let v = m.as_mut();
        let _ = v[(2, 0)];
    }));
    assert!(result.is_err());
}

#[test]
fn mat_mut_strides() {
    let mut m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let v = m.as_mut();
    assert_eq!(v.row_stride(), 1);
    assert_eq!(v.col_stride(), 2);
    assert_eq!(v.shape(), (2, 3));
    assert_eq!(v.size(), 6);
}

#[test]
fn col_iter_mut_exact_size() {
    let mut m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.col_iter_mut();
    assert_eq!(iter.len(), 3);
}

#[test]
fn row_iter_mut_exact_size() {
    let mut m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.row_iter_mut();
    assert_eq!(iter.len(), 2);
}

#[test]
fn mat_mut_enumerate() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    let items: Vec<_> = v.enumerate().collect();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0], ((0, 0), &1));
    assert_eq!(items[3], ((1, 1), &4));
}

#[test]
fn mat_mut_diag_iter() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    let diag: Vec<_> = v.diag_iter().collect();
    assert_eq!(diag, vec![&1, &4]);
}

#[test]
fn large_matrix_from_fn() {
    let n = 100;
    let m = Mat::from_fn(n, n, |i, j| i * n + j);
    assert_eq!(m.shape(), (n, n));
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(99, 99)], 99 * 100 + 99);
    assert_eq!(m[(50, 25)], 50 * 100 + 25);
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
fn from_rows_equals_from_row_major() {
    let a = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let b = Mat::from_vec_row(2, 3, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a, b);
}

#[test]
fn from_cols_equals_from_col_major() {
    let a = Mat::from_cols(&[&[1, 4], &[2, 5], &[3, 6]]);
    let b = Mat::from_vec_col(2, 3, vec![1, 4, 2, 5, 3, 6]);
    assert_eq!(a, b);
}

#[test]
fn from_blocks_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5], [6]];
    let c = mat![[7, 8]];
    let d = mat![[9]];

    let m = Mat::from_blocks(&[&[a.as_ref(), b.as_ref()], &[c.as_ref(), d.as_ref()]]);

    let expected = mat![[1, 2, 5], [3, 4, 6], [7, 8, 9]];
    assert_eq!(m, expected);
}

#[test]
fn from_blocks_single_block() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::from_blocks(&[&[a.as_ref()]]);
    assert_eq!(m, a);
}

#[test]
fn from_blocks_horizontal_concat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = Mat::from_blocks(&[&[a.as_ref(), b.as_ref()]]);
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8]];
    assert_eq!(m, expected);
}

#[test]
fn from_blocks_vertical_stack() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6]];
    let m = Mat::from_blocks(&[&[a.as_ref()], &[b.as_ref()]]);
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn from_blocks_empty() {
    let m: Mat<i32> = Mat::from_blocks(&[]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
fn from_blocks_empty_rows() {
    let m: Mat<i32> = Mat::from_blocks(&[&[], &[]]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
#[should_panic(expected = "Block row 1")]
fn from_blocks_inconsistent_block_count() {
    let a = mat![[1]];
    let b = mat![[2]];
    let c = mat![[3]];
    Mat::<i32>::from_blocks(&[&[a.as_ref(), b.as_ref()], &[c.as_ref()]]);
}

#[test]
#[should_panic(expected = "rows")]
fn from_blocks_inconsistent_row_heights() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5]];
    Mat::<i32>::from_blocks(&[&[a.as_ref(), b.as_ref()]]);
}

#[test]
#[should_panic(expected = "columns")]
fn from_blocks_inconsistent_col_widths() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6, 7]];
    Mat::<i32>::from_blocks(&[&[a.as_ref()], &[b.as_ref()]]);
}

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
fn block_macro_single_row() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = block![[a, b]];
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8]];
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
fn block_macro_with_identity() {
    let eye: Mat<i32> = Mat::identity(2);
    let z: Mat<i32> = Mat::zeros(2, 2);
    let m = block![[eye, z]];
    let expected = mat![[1, 0, 0, 0], [0, 1, 0, 0]];
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
fn is_empty_true_for_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_empty());
}

#[test]
fn is_empty_true_for_0xn() {
    let m: Mat<f64> = Mat::zeros(0, 3);
    assert!(m.is_empty());
}

#[test]
fn is_empty_false_for_nonempty() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_empty());
}

#[test]
fn is_square_true() {
    let m = mat![[1, 2], [3, 4]];
    assert!(m.is_square());
}

#[test]
fn is_square_false() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert!(!m.is_square());
}

#[test]
fn is_square_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_square());
}

#[test]
fn is_row_vector_true() {
    let r = row![1, 2, 3];
    assert!(r.is_row_vector());
}

#[test]
fn is_row_vector_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_row_vector());
}

#[test]
fn is_col_vector_true() {
    let c = col![1, 2, 3];
    assert!(c.is_col_vector());
}

#[test]
fn is_col_vector_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_col_vector());
}

#[test]
fn is_scalar_true() {
    let m = mat![[42]];
    assert!(m.is_scalar());
}

#[test]
fn is_scalar_false_row_vec() {
    let m = row![1, 2];
    assert!(!m.is_scalar());
}

#[test]
fn is_scalar_false_col_vec() {
    let m = col![1, 2];
    assert!(!m.is_scalar());
}

#[test]
fn shape_predicates_on_mat_ref() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.as_ref();
    assert!(!r.is_empty());
    assert!(r.is_square());
    assert!(!r.is_row_vector());
    assert!(!r.is_col_vector());
    assert!(!r.is_scalar());
}

#[test]
fn shape_predicates_on_mat_mut() {
    let mut m = col![10, 20, 30];
    let v = m.as_mut();
    assert!(!v.is_empty());
    assert!(!v.is_square());
    assert!(!v.is_row_vector());
    assert!(v.is_col_vector());
    assert!(!v.is_scalar());
}

#[test]
fn is_symmetric_identity() {
    let m: Mat<i32> = Mat::identity(3);
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_true() {
    let m = mat![[1, 2, 3], [2, 5, 6], [3, 6, 9]];
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_symmetric());
}

#[test]
fn is_symmetric_non_square() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert!(!m.is_symmetric());
}

#[test]
fn is_symmetric_1x1() {
    let m = mat![[7]];
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_symmetric());
}

#[test]
fn is_diagonal_true() {
    let m = Mat::diag(&[1, 2, 3]);
    assert!(m.is_diagonal());
}

#[test]
fn is_diagonal_identity() {
    let m: Mat<i32> = Mat::identity(3);
    assert!(m.is_diagonal());
}

#[test]
fn is_diagonal_false() {
    let m = mat![[1, 2], [0, 3]];
    assert!(!m.is_diagonal());
}

#[test]
fn is_diagonal_zeros() {
    let m: Mat<i32> = Mat::zeros(3, 3);
    assert!(m.is_diagonal());
}

#[test]
fn is_diagonal_non_square() {
    let m = mat![[1, 0, 0], [0, 2, 0]];
    assert!(m.is_diagonal());
}

#[test]
fn is_diagonal_non_square_false() {
    let m = mat![[1, 0, 1], [0, 2, 0]];
    assert!(!m.is_diagonal());
}

#[test]
fn is_upper_triangular_true() {
    let m = mat![[1, 2, 3], [0, 4, 5], [0, 0, 6]];
    assert!(m.is_upper_triangular());
}

#[test]
fn is_upper_triangular_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_upper_triangular());
}

#[test]
fn is_upper_triangular_identity() {
    let m: Mat<i32> = Mat::identity(3);
    assert!(m.is_upper_triangular());
}

#[test]
fn is_upper_triangular_non_square() {
    let m = mat![[1, 2, 3], [0, 4, 5]];
    assert!(m.is_upper_triangular());
}

#[test]
fn is_lower_triangular_true() {
    let m = mat![[1, 0, 0], [2, 3, 0], [4, 5, 6]];
    assert!(m.is_lower_triangular());
}

#[test]
fn is_lower_triangular_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_lower_triangular());
}

#[test]
fn is_lower_triangular_identity() {
    let m: Mat<i32> = Mat::identity(3);
    assert!(m.is_lower_triangular());
}

#[test]
fn is_lower_triangular_non_square() {
    let m = mat![[1, 0], [2, 3], [4, 5]];
    assert!(m.is_lower_triangular());
}

#[test]
fn is_identity_true() {
    let m: Mat<i32> = Mat::identity(4);
    assert!(m.is_identity());
}

#[test]
fn is_identity_false_off_diagonal() {
    let m = mat![[1, 1], [0, 1]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_false_diagonal() {
    let m = mat![[1, 0], [0, 2]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_non_square() {
    let m = mat![[1, 0, 0], [0, 1, 0]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_1x1() {
    let m = mat![[1]];
    assert!(m.is_identity());
}

#[test]
fn is_identity_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_identity());
}

#[test]
fn property_predicates_on_mat_ref() {
    let m: Mat<i32> = Mat::identity(3);
    let r = m.as_ref();
    assert!(r.is_symmetric());
    assert!(r.is_diagonal());
    assert!(r.is_upper_triangular());
    assert!(r.is_lower_triangular());
    assert!(r.is_identity());
}

#[test]
fn property_predicates_on_mat_mut() {
    let mut m: Mat<i32> = Mat::identity(2);
    let v = m.as_mut();
    assert!(v.is_symmetric());
    assert!(v.is_diagonal());
    assert!(v.is_upper_triangular());
    assert!(v.is_lower_triangular());
    assert!(v.is_identity());
}

#[test]
fn row_view() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let r = m.row(1);
    assert_eq!(r.shape(), (1, 3));
    assert_eq!(r[(0, 0)], 4);
    assert_eq!(r[(0, 1)], 5);
    assert_eq!(r[(0, 2)], 6);
}

#[test]
fn col_view() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let c = m.col(2);
    assert_eq!(c.shape(), (3, 1));
    assert_eq!(c[(0, 0)], 3);
    assert_eq!(c[(1, 0)], 6);
    assert_eq!(c[(2, 0)], 9);
}

#[test]
fn row_view_first_and_last() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let first = m.row(0);
    assert_eq!(first[(0, 0)], 1);
    assert_eq!(first[(0, 1)], 2);
    let last = m.row(2);
    assert_eq!(last[(0, 0)], 5);
    assert_eq!(last[(0, 1)], 6);
}

#[test]
fn col_view_first_and_last() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let first = m.col(0);
    assert_eq!(first[(0, 0)], 1);
    assert_eq!(first[(1, 0)], 4);
    let last = m.col(2);
    assert_eq!(last[(0, 0)], 3);
    assert_eq!(last[(1, 0)], 6);
}

#[test]
#[should_panic(expected = "Row index")]
fn row_view_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.row(2);
}

#[test]
#[should_panic(expected = "Column index")]
fn col_view_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.col(2);
}

#[test]
fn row_mut_view() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    {
        let mut r = m.row_mut(0);
        r[(0, 0)] = 10;
        r[(0, 1)] = 20;
        r[(0, 2)] = 30;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(0, 2)], 30);
    assert_eq!(m[(1, 0)], 4);
}

#[test]
fn col_mut_view() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    {
        let mut c = m.col_mut(1);
        c[(0, 0)] = 20;
        c[(1, 0)] = 40;
        c[(2, 0)] = 60;
    }
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(1, 1)], 40);
    assert_eq!(m[(2, 1)], 60);
    assert_eq!(m[(0, 0)], 1);
}

#[test]
fn diagonal_view_square() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let d = m.diagonal();
    assert_eq!(d.shape(), (3, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 5);
    assert_eq!(d[(2, 0)], 9);
}

#[test]
fn diagonal_view_wide() {
    let m = mat![[1, 2, 3, 4], [5, 6, 7, 8]];
    let d = m.diagonal();
    assert_eq!(d.shape(), (2, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 6);
}

#[test]
fn diagonal_view_tall() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let d = m.diagonal();
    assert_eq!(d.shape(), (2, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 4);
}

#[test]
fn diagonal_view_empty() {
    let m: Mat<i32> = Mat::new();
    let d = m.diagonal();
    assert_eq!(d.shape(), (0, 1));
}

#[test]
fn diagonal_mut_view() {
    let mut m = mat![[1, 0, 0], [0, 2, 0], [0, 0, 3]];
    {
        let mut d = m.diagonal_mut();
        d[(0, 0)] = 10;
        d[(1, 0)] = 20;
        d[(2, 0)] = 30;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(1, 1)], 20);
    assert_eq!(m[(2, 2)], 30);
    assert_eq!(m[(0, 1)], 0);
}

#[test]
fn submatrix_view() {
    let m = mat![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
    let sub = m.submatrix(1, 1, 2, 2);
    assert_eq!(sub.shape(), (2, 2));
    assert_eq!(sub[(0, 0)], 6);
    assert_eq!(sub[(0, 1)], 7);
    assert_eq!(sub[(1, 0)], 10);
    assert_eq!(sub[(1, 1)], 11);
}

#[test]
fn submatrix_full() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.submatrix(0, 0, 2, 2);
    assert_eq!(sub, m.as_ref());
}

#[test]
fn submatrix_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.submatrix(0, 0, 0, 0);
    assert_eq!(sub.shape(), (0, 0));
}

#[test]
#[should_panic(expected = "Row range")]
fn submatrix_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.submatrix(1, 0, 2, 1);
}

#[test]
#[should_panic(expected = "Column range")]
fn submatrix_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.submatrix(0, 1, 1, 2);
}

#[test]
fn submatrix_mut_view() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    {
        let mut sub = m.submatrix_mut(0, 1, 2, 2);
        sub[(0, 0)] = 20;
        sub[(1, 1)] = 60;
    }
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(1, 2)], 60);
    assert_eq!(m[(0, 0)], 1);
}

#[test]
fn rows_range_view() {
    let m = mat![[1, 2], [3, 4], [5, 6], [7, 8]];
    let sub = m.rows_range(1..3);
    assert_eq!(sub.shape(), (2, 2));
    assert_eq!(sub[(0, 0)], 3);
    assert_eq!(sub[(0, 1)], 4);
    assert_eq!(sub[(1, 0)], 5);
    assert_eq!(sub[(1, 1)], 6);
}

#[test]
fn cols_range_view() {
    let m = mat![[1, 2, 3, 4], [5, 6, 7, 8]];
    let sub = m.cols_range(1..3);
    assert_eq!(sub.shape(), (2, 2));
    assert_eq!(sub[(0, 0)], 2);
    assert_eq!(sub[(0, 1)], 3);
    assert_eq!(sub[(1, 0)], 6);
    assert_eq!(sub[(1, 1)], 7);
}

#[test]
fn rows_range_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.rows_range(1..1);
    assert_eq!(sub.shape(), (0, 2));
}

#[test]
fn cols_range_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.cols_range(0..0);
    assert_eq!(sub.shape(), (2, 0));
}

#[test]
fn rows_range_mut_view() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    {
        let mut sub = m.rows_range_mut(1..3);
        sub[(0, 0)] = 30;
        sub[(1, 1)] = 60;
    }
    assert_eq!(m[(1, 0)], 30);
    assert_eq!(m[(2, 1)], 60);
}

#[test]
fn cols_range_mut_view() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    {
        let mut sub = m.cols_range_mut(1..3);
        sub[(0, 0)] = 20;
        sub[(1, 1)] = 60;
    }
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(1, 2)], 60);
}

#[test]
fn split_at_row() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let (top, bottom) = m.split_at_row(1);
    assert_eq!(top.shape(), (1, 2));
    assert_eq!(top[(0, 0)], 1);
    assert_eq!(top[(0, 1)], 2);
    assert_eq!(bottom.shape(), (2, 2));
    assert_eq!(bottom[(0, 0)], 3);
    assert_eq!(bottom[(1, 1)], 6);
}

#[test]
fn split_at_col() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let (left, right) = m.split_at_col(2);
    assert_eq!(left.shape(), (2, 2));
    assert_eq!(left[(0, 0)], 1);
    assert_eq!(left[(1, 1)], 5);
    assert_eq!(right.shape(), (2, 1));
    assert_eq!(right[(0, 0)], 3);
    assert_eq!(right[(1, 0)], 6);
}

#[test]
fn split_at_row_edge_0() {
    let m = mat![[1, 2], [3, 4]];
    let (top, bottom) = m.split_at_row(0);
    assert_eq!(top.shape(), (0, 2));
    assert_eq!(bottom.shape(), (2, 2));
    assert_eq!(bottom[(0, 0)], 1);
}

#[test]
fn split_at_row_edge_nrows() {
    let m = mat![[1, 2], [3, 4]];
    let (top, bottom) = m.split_at_row(2);
    assert_eq!(top.shape(), (2, 2));
    assert_eq!(bottom.shape(), (0, 2));
    assert_eq!(top[(1, 1)], 4);
}

#[test]
fn split_at_col_edge_0() {
    let m = mat![[1, 2], [3, 4]];
    let (left, right) = m.split_at_col(0);
    assert_eq!(left.shape(), (2, 0));
    assert_eq!(right.shape(), (2, 2));
}

#[test]
fn split_at_col_edge_ncols() {
    let m = mat![[1, 2], [3, 4]];
    let (left, right) = m.split_at_col(2);
    assert_eq!(left.shape(), (2, 2));
    assert_eq!(right.shape(), (2, 0));
}

#[test]
#[should_panic(expected = "Split index")]
fn split_at_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.split_at_row(3);
}

#[test]
#[should_panic(expected = "Split index")]
fn split_at_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.split_at_col(3);
}

#[test]
fn split_at_row_mut_modifies_independently() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    {
        let (mut top, mut bottom) = m.split_at_row_mut(1);
        top[(0, 0)] = 10;
        bottom[(0, 0)] = 30;
        bottom[(1, 1)] = 60;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(1, 0)], 30);
    assert_eq!(m[(2, 1)], 60);
}

#[test]
fn split_at_col_mut_modifies_independently() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    {
        let (mut left, mut right) = m.split_at_col_mut(1);
        left[(0, 0)] = 10;
        right[(0, 0)] = 20;
        right[(1, 1)] = 60;
    }
    assert_eq!(m[(0, 0)], 10);
    assert_eq!(m[(0, 1)], 20);
    assert_eq!(m[(1, 2)], 60);
}

#[test]
fn transpose_view() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert_eq!(t[(0, 0)], 1);
    assert_eq!(t[(0, 1)], 4);
    assert_eq!(t[(1, 0)], 2);
    assert_eq!(t[(1, 1)], 5);
    assert_eq!(t[(2, 0)], 3);
    assert_eq!(t[(2, 1)], 6);
}

#[test]
fn transpose_view_square() {
    let m = mat![[1, 2], [3, 4]];
    let t = m.transpose();
    assert_eq!(t[(0, 0)], 1);
    assert_eq!(t[(0, 1)], 3);
    assert_eq!(t[(1, 0)], 2);
    assert_eq!(t[(1, 1)], 4);
}

#[test]
fn transpose_double_is_identity() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let tt = m.transpose().transpose();
    assert_eq!(tt, m.as_ref());
}

#[test]
fn transpose_empty() {
    let m: Mat<i32> = Mat::new();
    let t = m.transpose();
    assert_eq!(t.shape(), (0, 0));
}

#[test]
fn transpose_mut_view() {
    let mut m = mat![[1, 2], [3, 4]];
    {
        let mut t = m.transpose_mut();
        t[(0, 1)] = 30;
    }
    assert_eq!(m[(1, 0)], 30);
}

#[test]
fn reverse_rows_view() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let r = m.reverse_rows();
    assert_eq!(r.shape(), (3, 2));
    assert_eq!(r[(0, 0)], 5);
    assert_eq!(r[(0, 1)], 6);
    assert_eq!(r[(1, 0)], 3);
    assert_eq!(r[(1, 1)], 4);
    assert_eq!(r[(2, 0)], 1);
    assert_eq!(r[(2, 1)], 2);
}

#[test]
fn reverse_cols_view() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.reverse_cols();
    assert_eq!(r.shape(), (2, 3));
    assert_eq!(r[(0, 0)], 3);
    assert_eq!(r[(0, 1)], 2);
    assert_eq!(r[(0, 2)], 1);
    assert_eq!(r[(1, 0)], 6);
    assert_eq!(r[(1, 1)], 5);
    assert_eq!(r[(1, 2)], 4);
}

#[test]
fn reverse_rows_double_is_identity() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let rr = m.reverse_rows().reverse_rows();
    assert_eq!(rr, m.as_ref());
}

#[test]
fn reverse_cols_double_is_identity() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let rr = m.reverse_cols().reverse_cols();
    assert_eq!(rr, m.as_ref());
}

#[test]
fn reverse_rows_empty() {
    let m: Mat<i32> = Mat::new();
    let r = m.reverse_rows();
    assert_eq!(r.shape(), (0, 0));
}

#[test]
fn reverse_cols_empty() {
    let m: Mat<i32> = Mat::new();
    let r = m.reverse_cols();
    assert_eq!(r.shape(), (0, 0));
}

#[test]
fn reverse_rows_mut_view() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    {
        let mut r = m.reverse_rows_mut();
        r[(0, 0)] = 50;
    }
    assert_eq!(m[(2, 0)], 50);
}

#[test]
fn reverse_cols_mut_view() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    {
        let mut r = m.reverse_cols_mut();
        r[(0, 0)] = 30;
    }
    assert_eq!(m[(0, 2)], 30);
}

#[test]
fn transpose_reverse_rows_is_reverse_cols_transpose() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let a = m.transpose().reverse_rows().to_owned();
    let b = m.reverse_cols().transpose().to_owned();
    assert_eq!(a, b);
}

#[test]
fn tril_k0() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.tril(0);
    let expected = mat![[1, 0, 0], [4, 5, 0], [7, 8, 9]];
    assert_eq!(t, expected);
}

#[test]
fn tril_k1() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.tril(1);
    let expected = mat![[1, 2, 0], [4, 5, 6], [7, 8, 9]];
    assert_eq!(t, expected);
}

#[test]
fn tril_k_neg1() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.tril(-1);
    let expected = mat![[0, 0, 0], [4, 0, 0], [7, 8, 0]];
    assert_eq!(t, expected);
}

#[test]
fn triu_k0() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.triu(0);
    let expected = mat![[1, 2, 3], [0, 5, 6], [0, 0, 9]];
    assert_eq!(t, expected);
}

#[test]
fn triu_k1() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.triu(1);
    let expected = mat![[0, 2, 3], [0, 0, 6], [0, 0, 0]];
    assert_eq!(t, expected);
}

#[test]
fn triu_k_neg1() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let t = m.triu(-1);
    let expected = mat![[1, 2, 3], [4, 5, 6], [0, 8, 9]];
    assert_eq!(t, expected);
}

#[test]
fn tril_non_square() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.tril(0);
    let expected = mat![[1, 0, 0], [4, 5, 0]];
    assert_eq!(t, expected);
}

#[test]
fn triu_non_square() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let t = m.triu(0);
    let expected = mat![[1, 2], [0, 4], [0, 0]];
    assert_eq!(t, expected);
}

#[test]
fn take_rows_subset() {
    let m = mat![[1, 2], [3, 4], [5, 6], [7, 8]];
    let t = m.take_rows(&[0, 2, 3]);
    let expected = mat![[1, 2], [5, 6], [7, 8]];
    assert_eq!(t, expected);
}

#[test]
fn take_rows_reorder() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let t = m.take_rows(&[2, 0, 1]);
    let expected = mat![[5, 6], [1, 2], [3, 4]];
    assert_eq!(t, expected);
}

#[test]
fn take_rows_duplicate() {
    let m = mat![[1, 2], [3, 4]];
    let t = m.take_rows(&[0, 0, 1, 1]);
    let expected = mat![[1, 2], [1, 2], [3, 4], [3, 4]];
    assert_eq!(t, expected);
}

#[test]
fn take_rows_empty() {
    let m = mat![[1, 2], [3, 4]];
    let t = m.take_rows(&[]);
    assert_eq!(t.shape(), (0, 2));
}

#[test]
#[should_panic(expected = "Row index")]
fn take_rows_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.take_rows(&[0, 2]);
}

#[test]
fn take_cols_subset() {
    let m = mat![[1, 2, 3, 4], [5, 6, 7, 8]];
    let t = m.take_cols(&[1, 3]);
    let expected = mat![[2, 4], [6, 8]];
    assert_eq!(t, expected);
}

#[test]
fn take_cols_reorder() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.take_cols(&[2, 0, 1]);
    let expected = mat![[3, 1, 2], [6, 4, 5]];
    assert_eq!(t, expected);
}

#[test]
fn take_cols_empty() {
    let m = mat![[1, 2], [3, 4]];
    let t = m.take_cols(&[]);
    assert_eq!(t.shape(), (2, 0));
}

#[test]
#[should_panic(expected = "Column index")]
fn take_cols_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.take_cols(&[0, 2]);
}

#[test]
fn copy_from_full_matrix() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = Mat::zeros(2, 2);
    b.copy_from(a.as_ref());
    assert_eq!(b, a);
}

#[test]
fn copy_from_to_submatrix() {
    let src = mat![[10, 20], [30, 40]];
    let mut dst = Mat::zeros(4, 4);
    dst.submatrix_mut(1, 1, 2, 2).copy_from(src.as_ref());
    assert_eq!(dst[(0, 0)], 0);
    assert_eq!(dst[(1, 1)], 10);
    assert_eq!(dst[(1, 2)], 20);
    assert_eq!(dst[(2, 1)], 30);
    assert_eq!(dst[(2, 2)], 40);
    assert_eq!(dst[(3, 3)], 0);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn copy_from_shape_mismatch() {
    let a = mat![[1, 2, 3]];
    let mut b = Mat::zeros(2, 2);
    b.copy_from(a.as_ref());
}

#[test]
fn fill_entire_matrix() {
    let mut m = Mat::zeros(2, 3);
    m.fill(7);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(m[(i, j)], 7);
        }
    }
}

#[test]
fn fill_submatrix() {
    let mut m = Mat::zeros(3, 3);
    m.submatrix_mut(0, 0, 2, 2).fill(5);
    assert_eq!(m[(0, 0)], 5);
    assert_eq!(m[(0, 1)], 5);
    assert_eq!(m[(1, 0)], 5);
    assert_eq!(m[(1, 1)], 5);
    assert_eq!(m[(2, 2)], 0);
    assert_eq!(m[(0, 2)], 0);
}

#[test]
fn fill_with_fn_whole_matrix() {
    let mut m = Mat::zeros(2, 3);
    m.fill_with_fn(|i, j| (i * 10 + j) as i32);
    assert_eq!(m[(0, 0)], 0);
    assert_eq!(m[(0, 2)], 2);
    assert_eq!(m[(1, 0)], 10);
    assert_eq!(m[(1, 2)], 12);
}

#[test]
fn swap_rows_basic() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    m.swap_rows(0, 2);
    assert_eq!(m[(0, 0)], 5);
    assert_eq!(m[(0, 1)], 6);
    assert_eq!(m[(2, 0)], 1);
    assert_eq!(m[(2, 1)], 2);
    assert_eq!(m[(1, 0)], 3);
}

#[test]
fn swap_rows_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_rows(0, 0);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 0)], 3);
}

#[test]
#[should_panic(expected = "Row indices")]
fn swap_rows_out_of_bounds() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_rows(0, 2);
}

#[test]
fn swap_cols_basic() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    m.swap_cols(0, 2);
    assert_eq!(m[(0, 0)], 3);
    assert_eq!(m[(1, 0)], 6);
    assert_eq!(m[(0, 2)], 1);
    assert_eq!(m[(1, 2)], 4);
    assert_eq!(m[(0, 1)], 2);
}

#[test]
fn swap_cols_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_cols(0, 0);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
}

#[test]
#[should_panic(expected = "Column indices")]
fn swap_cols_out_of_bounds() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_cols(0, 2);
}

#[test]
fn mat_ref_row_col_views() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.as_ref();
    assert_eq!(r.row(0)[(0, 1)], 2);
    assert_eq!(r.col(2)[(1, 0)], 6);
}

#[test]
fn mat_ref_submatrix() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let sub = m.as_ref().submatrix(0, 1, 2, 2);
    assert_eq!(sub[(0, 0)], 2);
    assert_eq!(sub[(1, 1)], 6);
}

#[test]
fn mat_ref_transpose() {
    let m = mat![[1, 2], [3, 4]];
    let t = m.as_ref().transpose();
    assert_eq!(t[(0, 1)], 3);
    assert_eq!(t[(1, 0)], 2);
}

#[test]
fn mat_mut_row_col_readonly() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    let v = m.as_mut();
    assert_eq!(v.row(0)[(0, 1)], 2);
    assert_eq!(v.col(2)[(1, 0)], 6);
}

#[test]
fn mat_mut_submatrix_readonly() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let v = m.as_mut();
    let sub = v.submatrix(1, 0, 2, 2);
    assert_eq!(sub[(0, 0)], 4);
    assert_eq!(sub[(1, 1)], 8);
}

#[test]
fn mat_mut_transpose_readonly() {
    let mut m = mat![[1, 2], [3, 4]];
    let v = m.as_mut();
    let t = v.transpose();
    assert_eq!(t[(1, 0)], 2);
}

#[test]
fn mat_mut_tril() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let v = m.as_mut();
    let t = v.tril(0);
    let expected = mat![[1, 0, 0], [4, 5, 0], [7, 8, 9]];
    assert_eq!(t, expected);
}

#[test]
fn mat_mut_take_rows() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    let v = m.as_mut();
    let t = v.take_rows(&[2, 0]);
    let expected = mat![[5, 6], [1, 2]];
    assert_eq!(t, expected);
}

#[test]
fn submatrix_of_transpose() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let sub = m.transpose().submatrix(1, 0, 2, 2);
    assert_eq!(sub.shape(), (2, 2));
    assert_eq!(sub[(0, 0)], 2);
    assert_eq!(sub[(0, 1)], 5);
    assert_eq!(sub[(1, 0)], 3);
    assert_eq!(sub[(1, 1)], 6);
}

#[test]
fn row_of_reversed_matrix() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let r = m.reverse_rows().row(0);
    assert_eq!(r[(0, 0)], 5);
    assert_eq!(r[(0, 1)], 6);
}

#[test]
fn diagonal_of_transpose() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let d = m.transpose().diagonal();
    assert_eq!(d.shape(), (2, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 5);
}

#[test]
fn view_to_owned() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let sub = m.submatrix(0, 1, 2, 2);
    let owned = sub.to_owned();
    assert_eq!(owned, mat![[2, 3], [5, 6]]);
}

#[test]
fn reversed_view_to_owned() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let owned = m.reverse_rows().to_owned();
    assert_eq!(owned, mat![[5, 6], [3, 4], [1, 2]]);
}

#[test]
fn transposed_view_to_owned() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let owned = m.transpose().to_owned();
    assert_eq!(owned, mat![[1, 4], [2, 5], [3, 6]]);
}

#[test]
fn reshape_2x3_to_3x2() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.reshape(3, 2);
    assert_eq!(r.shape(), (3, 2));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 0)], 4);
    assert_eq!(r[(2, 0)], 2);
    assert_eq!(r[(0, 1)], 5);
    assert_eq!(r[(1, 1)], 3);
    assert_eq!(r[(2, 1)], 6);
}

#[test]
fn reshape_2x3_to_6x1() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.reshape(6, 1);
    assert_eq!(r.shape(), (6, 1));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 0)], 4);
    assert_eq!(r[(2, 0)], 2);
    assert_eq!(r[(3, 0)], 5);
    assert_eq!(r[(4, 0)], 3);
    assert_eq!(r[(5, 0)], 6);
}

#[test]
fn reshape_2x3_to_1x6() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.reshape(1, 6);
    assert_eq!(r.shape(), (1, 6));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(0, 1)], 4);
    assert_eq!(r[(0, 2)], 2);
    assert_eq!(r[(0, 3)], 5);
    assert_eq!(r[(0, 4)], 3);
    assert_eq!(r[(0, 5)], 6);
}

#[test]
fn reshape_same_shape() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.reshape(2, 2);
    assert_eq!(r, m);
}

#[test]
#[should_panic(expected = "Cannot reshape")]
fn reshape_mismatched_size() {
    let m = mat![[1, 2], [3, 4]];
    m.reshape(3, 3);
}

#[test]
fn flatten_2x3() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let f = m.flatten();
    assert_eq!(f.shape(), (6, 1));
    assert_eq!(f[(0, 0)], 1);
    assert_eq!(f[(1, 0)], 4);
    assert_eq!(f[(2, 0)], 2);
    assert_eq!(f[(3, 0)], 5);
}

#[test]
fn flatten_row_2x3() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let f = m.flatten_row();
    assert_eq!(f.shape(), (1, 6));
    assert_eq!(f[(0, 0)], 1);
    assert_eq!(f[(0, 1)], 4);
}

#[test]
fn flatten_1x1() {
    let m = mat![[42]];
    let f = m.flatten();
    assert_eq!(f.shape(), (1, 1));
    assert_eq!(f[(0, 0)], 42);
}

#[test]
fn to_col_vector_equals_flatten() {
    let m = mat![[1, 2], [3, 4]];
    assert_eq!(m.to_col_vector(), m.flatten());
}

#[test]
fn to_row_vector_equals_flatten_row() {
    let m = mat![[1, 2], [3, 4]];
    assert_eq!(m.to_row_vector(), m.flatten_row());
}

#[test]
fn vstack_basic() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6]];
    let m = Mat::vstack(&[a.as_ref(), b.as_ref()]);
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn vstack_three_matrices() {
    let a = mat![[1, 2]];
    let b = mat![[3, 4]];
    let c = mat![[5, 6]];
    let m = Mat::vstack(&[a.as_ref(), b.as_ref(), c.as_ref()]);
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn vstack_single() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::vstack(&[a.as_ref()]);
    assert_eq!(m, a);
}

#[test]
fn vstack_empty() {
    let m: Mat<i32> = Mat::vstack(&[]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
#[should_panic(expected = "columns")]
fn vstack_mismatched_cols() {
    let a = mat![[1, 2]];
    let b = mat![[3, 4, 5]];
    Mat::vstack(&[a.as_ref(), b.as_ref()]);
}

#[test]
fn hstack_basic() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5], [6]];
    let m = Mat::hstack(&[a.as_ref(), b.as_ref()]);
    let expected = mat![[1, 2, 5], [3, 4, 6]];
    assert_eq!(m, expected);
}

#[test]
fn hstack_three_matrices() {
    let a = mat![[1], [4]];
    let b = mat![[2], [5]];
    let c = mat![[3], [6]];
    let m = Mat::hstack(&[a.as_ref(), b.as_ref(), c.as_ref()]);
    let expected = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn hstack_single() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::hstack(&[a.as_ref()]);
    assert_eq!(m, a);
}

#[test]
fn hstack_empty() {
    let m: Mat<i32> = Mat::hstack(&[]);
    assert_eq!(m.shape(), (0, 0));
}

#[test]
#[should_panic(expected = "rows")]
fn hstack_mismatched_rows() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6]];
    Mat::hstack(&[a.as_ref(), b.as_ref()]);
}

#[test]
fn resize_larger() {
    let mut m = mat![[1, 2], [3, 4]];
    m.resize(3, 4, 0);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 1)], 4);
    assert_eq!(m[(2, 0)], 0);
    assert_eq!(m[(0, 2)], 0);
    assert_eq!(m[(2, 3)], 0);
}

#[test]
fn resize_smaller() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    m.resize(2, 2, 0);
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(1, 1)], 5);
}

#[test]
fn resize_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.resize(2, 2, 0);
    assert_eq!(m, mat![[1, 2], [3, 4]]);
}

#[test]
fn resize_add_rows_only() {
    let mut m = mat![[1, 2], [3, 4]];
    m.resize(4, 2, 0);
    assert_eq!(m.shape(), (4, 2));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 1)], 4);
    assert_eq!(m[(2, 0)], 0);
    assert_eq!(m[(3, 1)], 0);
}

#[test]
fn resize_add_cols_only() {
    let mut m = mat![[1, 2], [3, 4]];
    m.resize(2, 4, 9);
    assert_eq!(m.shape(), (2, 4));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 1)], 4);
    assert_eq!(m[(0, 2)], 9);
    assert_eq!(m[(1, 3)], 9);
}

#[test]
fn reserve_increases_capacity() {
    let mut m: Mat<i32> = Mat::zeros(2, 3);
    let before = m.as_slice().len();
    m.reserve(10);
    assert_eq!(m.as_slice().len(), before);
    assert_eq!(m.shape(), (2, 3));
}

#[test]
fn truncate_rows_and_cols() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    m.truncate(2, 2);
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(1, 0)], 4);
    assert_eq!(m[(1, 1)], 5);
}

#[test]
fn truncate_same_size() {
    let mut m = mat![[1, 2], [3, 4]];
    m.truncate(2, 2);
    assert_eq!(m, mat![[1, 2], [3, 4]]);
}

#[test]
#[should_panic(expected = "Cannot truncate")]
fn truncate_larger_panics() {
    let mut m = mat![[1, 2], [3, 4]];
    m.truncate(3, 2);
}

#[test]
fn insert_row_beginning() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(0, &[5, 6]);
    assert_eq!(r, mat![[5, 6], [1, 2], [3, 4]]);
}

#[test]
fn insert_row_middle() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(1, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [5, 6], [3, 4]]);
}

#[test]
fn insert_row_end() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(2, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [3, 4], [5, 6]]);
}

#[test]
#[should_panic(expected = "Row insert index")]
fn insert_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.insert_row(3, &[5, 6]);
}

#[test]
#[should_panic(expected = "Row length")]
fn insert_row_wrong_length() {
    let m = mat![[1, 2], [3, 4]];
    m.insert_row(0, &[5, 6, 7]);
}

#[test]
fn insert_col_beginning() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(0, &[5, 6]);
    assert_eq!(r, mat![[5, 1, 2], [6, 3, 4]]);
}

#[test]
fn insert_col_middle() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(1, &[5, 6]);
    assert_eq!(r, mat![[1, 5, 2], [3, 6, 4]]);
}

#[test]
fn insert_col_end() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(2, &[5, 6]);
    assert_eq!(r, mat![[1, 2, 5], [3, 4, 6]]);
}

#[test]
#[should_panic(expected = "Column insert index")]
fn insert_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.insert_col(3, &[5, 6]);
}

#[test]
#[should_panic(expected = "Column length")]
fn insert_col_wrong_length() {
    let m = mat![[1, 2], [3, 4]];
    m.insert_col(0, &[5]);
}

#[test]
fn remove_row_first() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(0), mat![[3, 4], [5, 6]]);
}

#[test]
fn remove_row_middle() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(1), mat![[1, 2], [5, 6]]);
}

#[test]
fn remove_row_last() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(2), mat![[1, 2], [3, 4]]);
}

#[test]
#[should_panic(expected = "Row index")]
fn remove_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.remove_row(2);
}

#[test]
fn remove_col_first() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(0), mat![[2, 3], [5, 6]]);
}

#[test]
fn remove_col_middle() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(1), mat![[1, 3], [4, 6]]);
}

#[test]
fn remove_col_last() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(2), mat![[1, 2], [4, 5]]);
}

#[test]
#[should_panic(expected = "Column index")]
fn remove_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.remove_col(2);
}

#[test]
fn append_row_basic() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.append_row(&[5, 6]);
    assert_eq!(r, mat![[1, 2], [3, 4], [5, 6]]);
}

#[test]
fn append_col_basic() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.append_col(&[5, 6]);
    assert_eq!(r, mat![[1, 2, 5], [3, 4, 6]]);
}

#[test]
fn mat_ref_reshape() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let r = m.as_ref().reshape(3, 2);
    assert_eq!(r.shape(), (3, 2));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 0)], 4);
}

#[test]
fn mat_ref_insert_row() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.as_ref().insert_row(1, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [5, 6], [3, 4]]);
}

#[test]
fn mat_mut_reshape() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    let v = m.as_mut();
    let r = v.reshape(3, 2);
    assert_eq!(r.shape(), (3, 2));
}

#[test]
fn mat_mut_flatten() {
    let mut m = mat![[1, 2], [3, 4]];
    let v = m.as_mut();
    let f = v.flatten();
    assert_eq!(f.shape(), (4, 1));
}

#[test]
fn reshape_transposed_view() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.transpose();
    let r = t.reshape(2, 3);
    assert_eq!(r.shape(), (2, 3));
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 0)], 2);
    assert_eq!(r[(0, 1)], 3);
    assert_eq!(r[(1, 1)], 4);
    assert_eq!(r[(0, 2)], 5);
    assert_eq!(r[(1, 2)], 6);
}

#[test]
fn add_matref_matref() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = a.as_ref() + b.as_ref();
    assert_eq!(c, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_ref_mat_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = &a + &b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_mat_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a + b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_mat_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a + &b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_ref_mat_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = &a + b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matref_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_ref() + &b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_ref_mat_matref() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = &a + b.as_ref();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_mat_matref() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a + b.as_ref();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_ref() + b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matmut_matref() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_mut() + b.as_ref();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matref_matmut() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let c = a.as_ref() + b.as_mut();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_ref_matmut_ref_matmut() {
    let mut a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let am = a.as_mut();
    let bm = b.as_mut();
    let c = &am + &bm;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn add_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3], [4, 5, 6]];
    let _ = &a + &b;
}

#[test]
fn sub_matref_matref() {
    let a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = a.as_ref() - b.as_ref();
    assert_eq!(c, mat![[9, 18], [27, 36]]);
}

#[test]
fn sub_ref_mat_ref_mat() {
    let a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = &a - &b;
    assert_eq!(c, mat![[9, 18], [27, 36]]);
}

#[test]
fn sub_mat_mat() {
    let a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = a - b;
    assert_eq!(c, mat![[9, 18], [27, 36]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn sub_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1], [2]];
    let _ = &a - &b;
}

#[test]
fn sub_matmut_matref() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = a.as_mut() - b.as_ref();
    assert_eq!(c, mat![[9, 18], [27, 36]]);
}

#[test]
fn mul_matref_scalar() {
    let a = mat![[1, 2], [3, 4]];
    let c = a.as_ref() * 10;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_ref_mat_scalar() {
    let a = mat![[1, 2], [3, 4]];
    let c = &a * 10;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_mat_scalar() {
    let a = mat![[1, 2], [3, 4]];
    let c = a * 10;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_matmut_scalar() {
    let mut a = mat![[1, 2], [3, 4]];
    let c = a.as_mut() * 10;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_scalar_matref() {
    let a = mat![[1, 2], [3, 4]];
    let c = 10 * a.as_ref();
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_scalar_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let c = 10 * &a;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_scalar_mat() {
    let a = mat![[1, 2], [3, 4]];
    let c = 10 * a;
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_scalar_matmut() {
    let mut a = mat![[1, 2], [3, 4]];
    let c = 10 * a.as_mut();
    assert_eq!(c, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_scalar_f64() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let c = 2.5 * &a;
    assert_eq!(c, mat![[2.5, 5.0], [7.5, 10.0]]);
}

#[test]
fn div_matref_scalar() {
    let a = mat![[10, 20], [30, 40]];
    let c = a.as_ref() / 10;
    assert_eq!(c, mat![[1, 2], [3, 4]]);
}

#[test]
fn div_ref_mat_scalar() {
    let a = mat![[10, 20], [30, 40]];
    let c = &a / 10;
    assert_eq!(c, mat![[1, 2], [3, 4]]);
}

#[test]
fn div_mat_scalar() {
    let a = mat![[10, 20], [30, 40]];
    let c = a / 10;
    assert_eq!(c, mat![[1, 2], [3, 4]]);
}

#[test]
fn div_matmut_scalar() {
    let mut a = mat![[10.0, 20.0], [30.0, 40.0]];
    let c = a.as_mut() / 10.0;
    assert_eq!(c, mat![[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn neg_matref() {
    let a = mat![[1, -2], [3, -4]];
    let c = -a.as_ref();
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn neg_ref_mat() {
    let a = mat![[1, -2], [3, -4]];
    let c = -&a;
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn neg_mat() {
    let a = mat![[1, -2], [3, -4]];
    let c = -a;
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn neg_matmut() {
    let mut a = mat![[1, -2], [3, -4]];
    let c = -a.as_mut();
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn neg_ref_matmut() {
    let mut a = mat![[1, -2], [3, -4]];
    let am = a.as_mut();
    let c = -&am;
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn add_assign_mat_ref_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    a += &b;
    assert_eq!(a, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_assign_mat_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    a += b;
    assert_eq!(a, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_assign_mat_matref() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    a += b.as_ref();
    assert_eq!(a, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_assign_matmut_matref() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    {
        let mut am = a.as_mut();
        am += b.as_ref();
    }
    assert_eq!(a, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_assign_matmut_ref_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    {
        let mut am = a.as_mut();
        am += &b;
    }
    assert_eq!(a, mat![[11, 22], [33, 44]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn add_assign_shape_mismatch() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3]];
    a += &b;
}

#[test]
fn sub_assign_mat_ref_mat() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    a -= &b;
    assert_eq!(a, mat![[9, 18], [27, 36]]);
}

#[test]
fn sub_assign_matmut_matref() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    {
        let mut am = a.as_mut();
        am -= b.as_ref();
    }
    assert_eq!(a, mat![[9, 18], [27, 36]]);
}

#[test]
fn mul_assign_mat_scalar() {
    let mut a = mat![[1, 2], [3, 4]];
    a *= 10;
    assert_eq!(a, mat![[10, 20], [30, 40]]);
}

#[test]
fn mul_assign_matmut_scalar() {
    let mut a = mat![[1, 2], [3, 4]];
    {
        let mut am = a.as_mut();
        am *= 10;
    }
    assert_eq!(a, mat![[10, 20], [30, 40]]);
}

#[test]
fn div_assign_mat_scalar() {
    let mut a = mat![[10, 20], [30, 40]];
    a /= 10;
    assert_eq!(a, mat![[1, 2], [3, 4]]);
}

#[test]
fn div_assign_matmut_scalar() {
    let mut a = mat![[10.0, 20.0], [30.0, 40.0]];
    {
        let mut am = a.as_mut();
        am /= 10.0;
    }
    assert_eq!(a, mat![[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn add_empty_matrices() {
    let a: Mat<i32> = Mat::zeros(0, 0);
    let b: Mat<i32> = Mat::zeros(0, 0);
    let c = &a + &b;
    assert_eq!(c.shape(), (0, 0));
}

#[test]
fn arithmetic_chain() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = &a + &b - &a;
    assert_eq!(c, b);
}

#[test]
fn scalar_ops_chain() {
    let a = mat![[2, 4], [6, 8]];
    let c = &a * 3 / 2;
    assert_eq!(c, mat![[3, 6], [9, 12]]);
}

#[test]
fn neg_then_add() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = -&a + &b;
    assert_eq!(c, mat![[4, 4], [4, 4]]);
}

#[test]
fn add_assign_then_mul_assign() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 1], [1, 1]];
    a += &b;
    a *= 10;
    assert_eq!(a, mat![[20, 30], [40, 50]]);
}

#[test]
fn ops_with_submatrix_view() {
    let a = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let b = mat![[10, 20], [30, 40]];
    let sub = a.submatrix(0, 0, 2, 2);
    let c = sub + b.as_ref();
    assert_eq!(c, mat![[11, 22], [34, 45]]);
}

#[test]
fn ops_with_transpose_view() {
    let a = mat![[1, 2], [3, 4]];
    let t = a.transpose();
    let b = mat![[10, 20], [30, 40]];
    let c = t + b.as_ref();
    assert_eq!(c, mat![[11, 23], [32, 44]]);
}

#[test]
fn scalar_mul_commutative() {
    let a = mat![[1, 2], [3, 4]];
    let c1 = &a * 5;
    let c2 = 5 * &a;
    assert_eq!(c1, c2);
}

#[test]
fn add_assign_on_submatrix_view() {
    let mut a = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let b = mat![[10, 20], [30, 40]];
    {
        let mut sub = a.as_mut().submatrix_mut(0, 0, 2, 2);
        sub += b.as_ref();
    }
    assert_eq!(a[(0, 0)], 11);
    assert_eq!(a[(0, 1)], 22);
    assert_eq!(a[(1, 0)], 34);
    assert_eq!(a[(1, 1)], 45);
    assert_eq!(a[(2, 2)], 9);
}

#[test]
fn mul_assign_on_submatrix_view() {
    let mut a = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    {
        let mut sub = a.as_mut().submatrix_mut(0, 0, 2, 2);
        sub *= 10;
    }
    assert_eq!(a[(0, 0)], 10);
    assert_eq!(a[(0, 1)], 20);
    assert_eq!(a[(1, 0)], 40);
    assert_eq!(a[(1, 1)], 50);
    assert_eq!(a[(2, 2)], 9);
}

#[test]
fn add_matmut_ref_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_mut() + &b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_ref_mat_matmut() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let c = &a + b.as_mut();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_mat_matmut() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let c = a + b.as_mut();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matmut_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_mut() + b;
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn sub_assign_mat_mat() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    a -= b;
    assert_eq!(a, mat![[9, 18], [27, 36]]);
}

#[test]
fn ops_1x1_scalar_like() {
    let a = mat![[5]];
    let b = mat![[3]];
    assert_eq!(&a + &b, mat![[8]]);
    assert_eq!(&a - &b, mat![[2]]);
    assert_eq!(&a * 2, mat![[10]]);
    assert_eq!(&a / 5, mat![[1]]);
    assert_eq!(-&a, mat![[-5]]);
}

#[test]
fn map_double() {
    let a = mat![[1, 2], [3, 4]];
    let b = a.map(|x| x * 2);
    assert_eq!(b, mat![[2, 4], [6, 8]]);
}

#[test]
fn map_type_change() {
    let a = mat![[1, 2], [3, 4]];
    let b: Mat<f64> = a.map(|x| *x as f64 + 0.5);
    assert_eq!(b, mat![[1.5, 2.5], [3.5, 4.5]]);
}

#[test]
fn map_empty() {
    let a: Mat<i32> = Mat::zeros(0, 0);
    let b = a.map(|x| x + 1);
    assert_eq!(b.shape(), (0, 0));
}

#[test]
fn map_on_matmut() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = a.as_mut().map(|x| x * 3);
    assert_eq!(b, mat![[3, 6], [9, 12]]);
}

#[test]
fn zip_map_add() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = a.zip_map(b.as_ref(), |x, y| x + y);
    assert_eq!(c, mat![[11, 22], [33, 44]]);
}

#[test]
fn zip_map_type_change() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2], [3, 5]];
    let c: Mat<bool> = a.zip_map(b.as_ref(), |x, y| x == y);
    assert_eq!(c, mat![[true, true], [true, false]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn zip_map_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3]];
    let _ = a.zip_map(b.as_ref(), |x, y| x + y);
}

#[test]
fn component_mul_basic() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.component_mul(b.as_ref());
    assert_eq!(c, mat![[5, 12], [21, 32]]);
}

#[test]
fn component_mul_matref() {
    let a = mat![[2.0, 3.0], [4.0, 5.0]];
    let b = mat![[0.5, 2.0], [0.25, 0.2]];
    let c = a.as_ref().component_mul(b.as_ref());
    assert_eq!(c, mat![[1.0, 6.0], [1.0, 1.0]]);
}

#[test]
fn component_div_basic() {
    let a = mat![[10, 20], [30, 40]];
    let b = mat![[2, 5], [6, 8]];
    let c = a.component_div(b.as_ref());
    assert_eq!(c, mat![[5, 4], [5, 5]]);
}

#[test]
fn component_div_f64() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let b = mat![[2.0, 4.0], [6.0, 8.0]];
    let c = a.as_ref().component_div(b.as_ref());
    assert_eq!(c, mat![[0.5, 0.5], [0.5, 0.5]]);
}

#[test]
fn abs_integers() {
    let a = mat![[1, -2], [-3, 4]];
    let b = a.abs();
    assert_eq!(b, mat![[1, 2], [3, 4]]);
}

#[test]
fn abs_f64() {
    let a = mat![[-1.5, 2.5], [3.0, -4.0]];
    let b = a.abs();
    assert_eq!(b, mat![[1.5, 2.5], [3.0, 4.0]]);
}

#[test]
fn abs_on_matmut() {
    let mut a = mat![[-1, 2], [-3, 4]];
    let b = a.as_mut().abs();
    assert_eq!(b, mat![[1, 2], [3, 4]]);
}

#[test]
fn signum_integers() {
    let a = mat![[5, -3], [0, 7]];
    let b = a.signum();
    assert_eq!(b, mat![[1, -1], [0, 1]]);
}

#[test]
fn signum_f64() {
    let a = mat![[-2.5, 0.0], [3.0, -0.0]];
    let b = a.signum();
    assert_eq!(b[(0, 0)], -1.0);
    assert_eq!(b[(1, 0)], 1.0);
}

#[test]
fn clamp_basic() {
    let a = mat![[1, 5, 3], [8, 2, 7]];
    let b = a.clamp(2, 6);
    assert_eq!(b, mat![[2, 5, 3], [6, 2, 6]]);
}

#[test]
fn clamp_f64() {
    let a = mat![[-1.0, 0.5], [1.5, 2.0]];
    let b = a.clamp(0.0, 1.0);
    assert_eq!(b, mat![[0.0, 0.5], [1.0, 1.0]]);
}

#[test]
fn clamp_all_within() {
    let a = mat![[3, 4], [5, 6]];
    let b = a.clamp(1, 10);
    assert_eq!(b, a);
}

#[test]
fn pow_basic() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let b = a.pow(2.0);
    assert_eq!(b, mat![[1.0, 4.0], [9.0, 16.0]]);
}

#[test]
fn pow_sqrt_via_half() {
    let a = mat![[4.0, 9.0], [16.0, 25.0]];
    let b = a.pow(0.5);
    assert_eq!(b, mat![[2.0, 3.0], [4.0, 5.0]]);
}

#[test]
fn sqrt_basic() {
    let a = mat![[1.0, 4.0], [9.0, 16.0]];
    let b = a.sqrt();
    assert_eq!(b, mat![[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn cbrt_basic() {
    let a = mat![[8.0, 27.0], [64.0, 125.0]];
    let b = a.cbrt();
    assert_eq!(b, mat![[2.0, 3.0], [4.0, 5.0]]);
}

#[test]
fn exp_ln_roundtrip() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = a.exp().as_ref().ln();
    for j in 0..2 {
        for i in 0..2 {
            assert!((b[(i, j)] - a[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn log10_basic() {
    let a = mat![[1.0, 10.0], [100.0, 1000.0]];
    let b = a.log10();
    assert_eq!(b, mat![[0.0, 1.0], [2.0, 3.0]]);
}

#[test]
fn log2_basic() {
    let a = mat![[1.0, 2.0], [4.0, 8.0]];
    let b = a.log2();
    assert_eq!(b, mat![[0.0, 1.0], [2.0, 3.0]]);
}

#[test]
fn sin_cos_identity() {
    let a = mat![
        [0.0_f64, std::f64::consts::FRAC_PI_6],
        [std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2]
    ];
    let s = a.sin();
    let c = a.cos();
    for j in 0..2 {
        for i in 0..2 {
            let sum = s[(i, j)] * s[(i, j)] + c[(i, j)] * c[(i, j)];
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn tan_basic() {
    let a = mat![[0.0_f64]];
    let b = a.tan();
    assert!((b[(0, 0)]).abs() < 1e-10);
}

#[test]
fn asin_acos_atan() {
    let a = mat![[0.0_f64, 0.5], [-0.5, 1.0]];
    let s = a.as_ref().asin();
    let c = a.as_ref().acos();
    for j in 0..2 {
        for i in 0..2 {
            assert!((s[(i, j)] + c[(i, j)] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }
    }
    let t = a.as_ref().atan();
    assert!((t[(0, 0)]).abs() < 1e-10);
}

#[test]
fn sinh_cosh_identity() {
    let a = mat![[0.5_f64, 1.0], [1.5, 2.0]];
    let sh = a.sinh();
    let ch = a.cosh();
    for j in 0..2 {
        for i in 0..2 {
            let diff = ch[(i, j)] * ch[(i, j)] - sh[(i, j)] * sh[(i, j)];
            assert!((diff - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn tanh_range() {
    let a = mat![[-10.0_f64, 0.0], [0.5, 10.0]];
    let b = a.tanh();
    for j in 0..2 {
        for i in 0..2 {
            assert!(b[(i, j)] >= -1.0 && b[(i, j)] <= 1.0);
        }
    }
    assert!((b[(0, 1)]).abs() < 1e-10);
}

#[test]
fn asinh_acosh_atanh() {
    let val = 2.0_f64;
    let a = mat![[val]];
    let b = a.as_ref().sinh().as_ref().asinh();
    assert!((b[(0, 0)] - val).abs() < 1e-10);

    let a2 = mat![[val]];
    let b2 = a2.as_ref().cosh().as_ref().acosh();
    assert!((b2[(0, 0)] - val).abs() < 1e-10);

    let a3 = mat![[0.5_f64]];
    let b3 = a3.as_ref().tanh().as_ref().atanh();
    assert!((b3[(0, 0)] - 0.5).abs() < 1e-10);
}

#[test]
fn ceil_floor_round() {
    let a = mat![[1.2, 2.5], [3.7, -1.3]];
    let c = a.ceil();
    assert_eq!(c, mat![[2.0, 3.0], [4.0, -1.0]]);
    let f = a.floor();
    assert_eq!(f, mat![[1.0, 2.0], [3.0, -2.0]]);
    let r = a.round();
    assert_eq!(r, mat![[1.0, 3.0], [4.0, -1.0]]);
}

#[test]
fn math_on_view() {
    let a = mat![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]];
    let sub = a.submatrix(0, 0, 2, 2);
    let b = sub.sqrt();
    assert_eq!(b, mat![[1.0, 2.0], [4.0, 5.0]]);
}

#[test]
fn map_on_transpose() {
    let a = mat![[1, 2], [3, 4]];
    let b = a.transpose().map(|x| x * 10);
    assert_eq!(b, mat![[10, 30], [20, 40]]);
}

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[test]
fn complex_construct_and_equality() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m[(0, 0)], c(1.0, 2.0));
    assert_eq!(m[(1, 1)], c(7.0, 8.0));
}

#[test]
fn complex_re_im() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    let re = m.re();
    let im = m.im();
    assert_eq!(re, mat![[1.0, 3.0], [5.0, 7.0]]);
    assert_eq!(im, mat![[2.0, 4.0], [6.0, 8.0]]);
}

#[test]
fn complex_re_im_on_ref() {
    let m = mat![[c(1.0, 2.0), c(3.0, -1.0)]];
    let r = m.as_ref();
    assert_eq!(r.re(), mat![[1.0, 3.0]]);
    assert_eq!(r.im(), mat![[2.0, -1.0]]);
}

#[test]
fn complex_re_im_on_mut() {
    let mut m = mat![[c(1.0, 2.0)]];
    let mm = m.as_mut();
    assert_eq!(mm.re(), mat![[1.0]]);
    assert_eq!(mm.im(), mat![[2.0]]);
}

#[test]
fn complex_conj() {
    let m = mat![[c(1.0, 2.0), c(3.0, -4.0)], [c(0.0, 1.0), c(-1.0, 0.0)]];
    let conj = m.conj();
    assert_eq!(
        conj,
        mat![[c(1.0, -2.0), c(3.0, 4.0)], [c(0.0, -1.0), c(-1.0, 0.0)]]
    );
}

#[test]
fn complex_conj_transpose() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    let ct = m.conj_transpose();
    assert_eq!(ct.shape(), (2, 2));
    assert_eq!(ct[(0, 0)], c(1.0, -2.0));
    assert_eq!(ct[(0, 1)], c(5.0, -6.0));
    assert_eq!(ct[(1, 0)], c(3.0, -4.0));
    assert_eq!(ct[(1, 1)], c(7.0, -8.0));
}

#[test]
fn complex_conj_transpose_nonsquare() {
    let m = mat![[c(1.0, 1.0), c(2.0, 2.0), c(3.0, 3.0)]];
    let ct = m.conj_transpose();
    assert_eq!(ct.shape(), (3, 1));
    assert_eq!(ct[(0, 0)], c(1.0, -1.0));
    assert_eq!(ct[(1, 0)], c(2.0, -2.0));
    assert_eq!(ct[(2, 0)], c(3.0, -3.0));
}

#[test]
fn complex_is_hermitian_true() {
    let m = mat![
        [c(1.0, 0.0), c(2.0, 3.0), c(4.0, -1.0)],
        [c(2.0, -3.0), c(5.0, 0.0), c(6.0, 7.0)],
        [c(4.0, 1.0), c(6.0, -7.0), c(9.0, 0.0)]
    ];
    assert!(m.is_hermitian());
}

#[test]
fn complex_is_hermitian_false_offdiag() {
    let m = mat![[c(1.0, 0.0), c(2.0, 3.0)], [c(2.0, 3.0), c(5.0, 0.0)]];
    assert!(!m.is_hermitian());
}

#[test]
fn complex_is_hermitian_false_diag() {
    let m = mat![[c(1.0, 1.0), c(2.0, 3.0)], [c(2.0, -3.0), c(5.0, 0.0)]];
    assert!(!m.is_hermitian());
}

#[test]
fn complex_is_hermitian_nonsquare() {
    let m = mat![[c(1.0, 0.0), c(2.0, 0.0)]];
    assert!(!m.is_hermitian());
}

#[test]
fn complex_is_hermitian_1x1() {
    let m = mat![[c(3.0, 0.0)]];
    assert!(m.is_hermitian());
    let m2 = mat![[c(3.0, 1.0)]];
    assert!(!m2.is_hermitian());
}

#[test]
fn complex_norm() {
    let m = mat![[c(3.0, 4.0), c(0.0, 1.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    let n = m.norm();
    assert!((n[(0, 0)] - 5.0).abs() < 1e-10);
    assert!((n[(0, 1)] - 1.0).abs() < 1e-10);
    assert!((n[(1, 0)] - 1.0).abs() < 1e-10);
    assert!((n[(1, 1)] - 0.0).abs() < 1e-10);
}

#[test]
fn complex_norm_sqr() {
    let m = mat![[c(3.0, 4.0), c(1.0, 1.0)]];
    let ns = m.norm_sqr();
    assert!((ns[(0, 0)] - 25.0).abs() < 1e-10);
    assert!((ns[(0, 1)] - 2.0).abs() < 1e-10);
}

#[test]
fn complex_arg() {
    let m = mat![[c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0), c(0.0, -1.0)]];
    let a = m.arg();
    assert!((a[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((a[(0, 1)] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((a[(0, 2)] - std::f64::consts::PI).abs() < 1e-10);
    assert!((a[(0, 3)] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn complex_powc() {
    let m = mat![[c(1.0, 0.0), c(0.0, 1.0)]];
    let result = m.powc(c(2.0, 0.0));
    assert!((result[(0, 0)] - c(1.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 1)] - c(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_powf() {
    let m = mat![[c(1.0, 0.0), c(4.0, 0.0)]];
    let result = m.powf(0.5);
    assert!((result[(0, 0)] - c(1.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 1)] - c(2.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_log() {
    let m = mat![[c(1.0, 0.0)]];
    let result = m.log(std::f64::consts::E);
    assert!((result[(0, 0)] - c(0.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_add_sub() {
    let a = mat![[c(1.0, 2.0), c(3.0, 4.0)]];
    let b = mat![[c(5.0, 6.0), c(7.0, 8.0)]];
    let sum = &a + &b;
    assert_eq!(sum, mat![[c(6.0, 8.0), c(10.0, 12.0)]]);
    let diff = &a - &b;
    assert_eq!(diff, mat![[c(-4.0, -4.0), c(-4.0, -4.0)]]);
}

#[test]
fn complex_scalar_mul() {
    let a = mat![[c(1.0, 2.0), c(3.0, 4.0)]];
    let result = &a * c(2.0, 0.0);
    assert_eq!(result, mat![[c(2.0, 4.0), c(6.0, 8.0)]]);
    let result2 = &a * c(0.0, 1.0);
    assert_eq!(result2, mat![[c(-2.0, 1.0), c(-4.0, 3.0)]]);
}

#[test]
fn complex_scalar_lmul() {
    let a = mat![[c(1.0, 2.0), c(3.0, 4.0)]];
    let result = c(2.0, 0.0) * &a;
    assert_eq!(result, mat![[c(2.0, 4.0), c(6.0, 8.0)]]);
}

#[test]
fn complex_neg() {
    let a = mat![[c(1.0, 2.0), c(-3.0, 4.0)]];
    let result = -&a;
    assert_eq!(result, mat![[c(-1.0, -2.0), c(3.0, -4.0)]]);
}

#[test]
fn complex_component_mul() {
    let a = mat![[c(1.0, 0.0), c(0.0, 1.0)]];
    let b = mat![[c(0.0, 1.0), c(0.0, 1.0)]];
    let result = a.component_mul(b.as_ref());
    assert_eq!(result, mat![[c(0.0, 1.0), c(-1.0, 0.0)]]);
}

#[test]
fn complex_assign_ops() {
    let mut a = mat![[c(1.0, 2.0), c(3.0, 4.0)]];
    let b = mat![[c(1.0, 1.0), c(1.0, 1.0)]];
    a += &b;
    assert_eq!(a, mat![[c(2.0, 3.0), c(4.0, 5.0)]]);
    a -= &b;
    assert_eq!(a, mat![[c(1.0, 2.0), c(3.0, 4.0)]]);
    a *= c(2.0, 0.0);
    assert_eq!(a, mat![[c(2.0, 4.0), c(6.0, 8.0)]]);
}

#[test]
fn complex_map_sqrt() {
    let m = mat![[c(0.0, 0.0), c(4.0, 0.0), c(-1.0, 0.0)]];
    let result = m.map(|x| x.sqrt());
    assert!((result[(0, 0)] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 1)] - c(2.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 2)] - c(0.0, 1.0)).norm() < 1e-10);
}

#[test]
fn complex_map_exp_ln_roundtrip() {
    let m = mat![[c(1.0, 2.0), c(-1.0, 0.5)]];
    let result = m.map(|x| x.exp()).map(|x| x.ln());
    assert!((result[(0, 0)] - c(1.0, 2.0)).norm() < 1e-10);
    assert!((result[(0, 1)] - c(-1.0, 0.5)).norm() < 1e-10);
}

#[test]
fn complex_map_sin_cos_identity() {
    let z = c(1.0, 2.0);
    let m = mat![[z]];
    let sin2 = m.map(|x| x.sin() * x.sin());
    let cos2 = m.map(|x| x.cos() * x.cos());
    let sum = &sin2 + &cos2;
    assert!((sum[(0, 0)] - c(1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_map_trig() {
    let m = mat![[c(0.0, 0.0)]];
    let s = m.map(|x| x.sin());
    let co = m.map(|x| x.cos());
    assert!((s[(0, 0)] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((co[(0, 0)] - c(1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_zeros_ones_identity() {
    let z: Mat<Complex<f64>> = Mat::zeros(2, 2);
    assert_eq!(z[(0, 0)], c(0.0, 0.0));
    let o: Mat<Complex<f64>> = Mat::ones(2, 2);
    assert_eq!(o[(0, 0)], c(1.0, 0.0));
    let id: Mat<Complex<f64>> = Mat::identity(2);
    assert_eq!(id[(0, 0)], c(1.0, 0.0));
    assert_eq!(id[(0, 1)], c(0.0, 0.0));
    assert_eq!(id[(1, 1)], c(1.0, 0.0));
}

#[test]
fn complex_transpose() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    let t = m.transpose();
    assert_eq!(t[(0, 0)], c(1.0, 2.0));
    assert_eq!(t[(0, 1)], c(5.0, 6.0));
    assert_eq!(t[(1, 0)], c(3.0, 4.0));
    assert_eq!(t[(1, 1)], c(7.0, 8.0));
}

#[test]
fn complex_is_symmetric() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(3.0, 4.0), c(5.0, 6.0)]];
    assert!(m.is_symmetric());
}

#[test]
fn complex_conj_on_view() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    let sub = m.submatrix(0, 0, 1, 2);
    let conj = sub.conj();
    assert_eq!(conj, mat![[c(1.0, -2.0), c(3.0, -4.0)]]);
}

#[test]
fn complex_norm_on_view() {
    let m = mat![[c(3.0, 4.0), c(0.0, 0.0)], [c(0.0, 0.0), c(5.0, 12.0)]];
    let sub = m.submatrix(1, 1, 1, 1);
    let n = sub.norm();
    assert!((n[(0, 0)] - 13.0).abs() < 1e-10);
}

#[test]
fn complex_empty_matrix() {
    let m: Mat<Complex<f64>> = Mat::new();
    let re = m.re();
    assert_eq!(re.shape(), (0, 0));
    let conj = m.conj();
    assert_eq!(conj.shape(), (0, 0));
}

#[test]
fn complex_scalar_div() {
    let a = mat![[c(4.0, 2.0), c(6.0, 0.0)]];
    let result = &a / c(2.0, 0.0);
    assert_eq!(result, mat![[c(2.0, 1.0), c(3.0, 0.0)]]);
}

#[test]
fn complex_div_assign() {
    let mut a = mat![[c(4.0, 2.0)]];
    a /= c(2.0, 0.0);
    assert_eq!(a, mat![[c(2.0, 1.0)]]);
}
