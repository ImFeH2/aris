use super::*;
use crate::{col, mat, row};

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

// --- Stride and Storage ---

#[test]
fn col_major_storage_layout() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    assert_eq!(m.row_stride(), 1);
    assert_eq!(m.col_stride(), 2);
    assert_eq!(m.as_slice(), &[1, 4, 2, 5, 3, 6]);
}

// --- Indexing ---

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

// --- as_slice ---

#[test]
fn as_slice_mut_modifies_data() {
    let mut m = Mat::zeros(2, 2);
    m.as_slice_mut()[0] = 7.0;
    assert_eq!(m[(0, 0)], 7.0);
}

// --- Views: as_ref / as_mut ---

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
    let mut v = m.as_mut();
    v[(0, 1)] = 20;
    drop(v);
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
    let mut v = m.as_mut();
    assert_eq!(v.get(1, 0), Some(&3));
    assert_eq!(v.get(2, 0), None);
    *v.get_mut(1, 1).unwrap() = 44;
    drop(v);
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
    let mut v = m.as_mut();
    let mut rb = v.rb_mut();
    rb[(0, 0)] = 10;
    drop(rb);
    drop(v);
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

// --- Iterators ---

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

// --- Traits ---

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

// --- Display / Debug ---

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

// --- Macros ---

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

// --- Edge Cases ---

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

// --- Block Matrix (from_blocks / mat!{{...}}) ---

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
fn mat_block_macro_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5], [6]];
    let c = mat![[7, 8]];
    let d = mat![[9]];

    let m = mat! {{a, b}, {c, d}};
    let expected = mat![[1, 2, 5], [3, 4, 6], [7, 8, 9]];
    assert_eq!(m, expected);
}

#[test]
fn mat_block_macro_single_row() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = mat! {{a, b}};
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8]];
    assert_eq!(m, expected);
}

#[test]
fn mat_block_macro_single_column() {
    let a = mat![[1, 2]];
    let b = mat![[3, 4]];
    let c = mat![[5, 6]];
    let m = mat! {{a}, {b}, {c}};
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
}

#[test]
fn mat_block_macro_with_identity() {
    let eye: Mat<i32> = Mat::identity(2);
    let z: Mat<i32> = Mat::zeros(2, 2);
    let m = mat! {{eye, z}};
    let expected = mat![[1, 0, 0, 0], [0, 1, 0, 0]];
    assert_eq!(m, expected);
}

#[test]
fn mat_block_macro_trailing_comma() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = mat! {{a, b,}, {b, a,},};
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8], [5, 6, 1, 2], [7, 8, 3, 4]];
    assert_eq!(m, expected);
}
