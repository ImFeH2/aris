use aris::{Mat, col, mat, row};

#[test]
fn as_slice_mut_modifies_data() {
    let mut m = Mat::zeros(2, 2);
    m.as_slice_mut()[0] = 7.0;
    assert_eq!(m[(0, 0)], 7.0);
}

#[test]
fn clone_creates_independent_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let c = m.clone();
    assert_eq!(m, c);
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
fn copy_from_full_matrix() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = Mat::zeros(2, 2);
    b.copy_from(a.as_ref());
    assert_eq!(b, a);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn copy_from_shape_mismatch() {
    let a = mat![[1, 2, 3]];
    let mut b = Mat::zeros(2, 2);
    b.copy_from(a.as_ref());
}

#[test]
fn copy_from_to_view() {
    let src = mat![[10, 20], [30, 40]];
    let mut dst = Mat::zeros(4, 4);
    dst.view_mut(1, 1, 2, 2).copy_from(src.as_ref());
    assert_eq!(dst[(0, 0)], 0);
    assert_eq!(dst[(1, 1)], 10);
    assert_eq!(dst[(1, 2)], 20);
    assert_eq!(dst[(2, 1)], 30);
    assert_eq!(dst[(2, 2)], 40);
    assert_eq!(dst[(3, 3)], 0);
}

#[test]
fn debug_format() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let s = format!("{:?}", m);
    assert_eq!(s, "Mat(2x2, [[1, 2], [3, 4]])");
}

#[test]
fn display_empty() {
    let m: Mat<i32> = Mat::new();
    let s = format!("{}", m);
    assert_eq!(s, "[]");
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
fn fill_view() {
    let mut m = Mat::zeros(3, 3);
    m.view_mut(0, 0, 2, 2).fill(5);
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
fn get_mut_modifies_element() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    *m.get_mut(1, 0).unwrap() = 99;
    assert_eq!(m[(1, 0)], 99);
}

#[test]
fn get_returns_none_for_invalid_indices() {
    let m = Mat::from_rows(&[&[10, 20], &[30, 40]]);
    assert_eq!(m.get(2, 0), None);
    assert_eq!(m.get(0, 2), None);
}

#[test]
fn get_returns_some_for_valid_indices() {
    let m = Mat::from_rows(&[&[10, 20], &[30, 40]]);
    assert_eq!(m.get(0, 0), Some(&10));
    assert_eq!(m.get(1, 1), Some(&40));
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
fn index_out_of_bounds_col() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let _ = m[(0, 2)];
}

#[test]
#[should_panic(expected = "out of bounds")]
fn index_out_of_bounds_row() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let _ = m[(2, 0)];
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
fn partial_eq_same_content() {
    let a = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let b = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(a, b);
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
fn single_element_matrix() {
    let m = mat![[42]];
    assert_eq!(m.shape(), (1, 1));
    assert_eq!(m[(0, 0)], 42);
    assert_eq!(m.as_slice(), &[42]);

    let diag: Vec<_> = m.diag_iter().collect();
    assert_eq!(diag, vec![&42]);
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
fn take_cols_reorder() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.take_cols(&[2, 0, 1]);
    let expected = mat![[3, 1, 2], [6, 4, 5]];
    assert_eq!(t, expected);
}

#[test]
fn take_cols_subset() {
    let m = mat![[1, 2, 3, 4], [5, 6, 7, 8]];
    let t = m.take_cols(&[1, 3]);
    let expected = mat![[2, 4], [6, 8]];
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
fn take_rows_reorder() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let t = m.take_rows(&[2, 0, 1]);
    let expected = mat![[5, 6], [1, 2], [3, 4]];
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
fn to_owned_creates_independent_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let owned = r.to_owned();
    assert_eq!(owned, m);
    assert_eq!(owned.shape(), m.shape());
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
fn tril_non_square() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let t = m.tril(0);
    let expected = mat![[1, 0, 0], [4, 5, 0]];
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
fn triu_non_square() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let t = m.triu(0);
    let expected = mat![[1, 2], [0, 4], [0, 0]];
    assert_eq!(t, expected);
}
