use aris::{Mat, mat};

#[test]
fn append_col_basic() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.append_col(&[5, 6]);
    assert_eq!(r, mat![[1, 2, 5], [3, 4, 6]]);
}

#[test]
fn append_row_basic() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.append_row(&[5, 6]);
    assert_eq!(r, mat![[1, 2], [3, 4], [5, 6]]);
}

#[test]
fn flatten_1x1() {
    let m = mat![[42]];
    let f = m.flatten();
    assert_eq!(f.shape(), (1, 1));
    assert_eq!(f[(0, 0)], 42);
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
fn from_blocks_horizontal_concat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let m = Mat::from_blocks(&[&[a.as_ref(), b.as_ref()]]);
    let expected = mat![[1, 2, 5, 6], [3, 4, 7, 8]];
    assert_eq!(m, expected);
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
#[should_panic(expected = "columns")]
fn from_blocks_inconsistent_col_widths() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6, 7]];
    Mat::<i32>::from_blocks(&[&[a.as_ref()], &[b.as_ref()]]);
}

#[test]
#[should_panic(expected = "rows")]
fn from_blocks_inconsistent_row_heights() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5]];
    Mat::<i32>::from_blocks(&[&[a.as_ref(), b.as_ref()]]);
}

#[test]
fn from_blocks_single_block() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::from_blocks(&[&[a.as_ref()]]);
    assert_eq!(m, a);
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
fn hstack_basic() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5], [6]];
    let m = Mat::hstack(&[a.as_ref(), b.as_ref()]);
    let expected = mat![[1, 2, 5], [3, 4, 6]];
    assert_eq!(m, expected);
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
fn hstack_single() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::hstack(&[a.as_ref()]);
    assert_eq!(m, a);
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
fn insert_col_beginning() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(0, &[5, 6]);
    assert_eq!(r, mat![[5, 1, 2], [6, 3, 4]]);
}

#[test]
fn insert_col_end() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(2, &[5, 6]);
    assert_eq!(r, mat![[1, 2, 5], [3, 4, 6]]);
}

#[test]
fn insert_col_middle() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_col(1, &[5, 6]);
    assert_eq!(r, mat![[1, 5, 2], [3, 6, 4]]);
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
fn insert_row_beginning() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(0, &[5, 6]);
    assert_eq!(r, mat![[5, 6], [1, 2], [3, 4]]);
}

#[test]
fn insert_row_end() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(2, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [3, 4], [5, 6]]);
}

#[test]
fn insert_row_middle() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.insert_row(1, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [5, 6], [3, 4]]);
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
fn remove_col_first() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(0), mat![[2, 3], [5, 6]]);
}

#[test]
fn remove_col_last() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(2), mat![[1, 2], [4, 5]]);
}

#[test]
fn remove_col_middle() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert_eq!(m.remove_col(1), mat![[1, 3], [4, 6]]);
}

#[test]
#[should_panic(expected = "Column index")]
fn remove_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.remove_col(2);
}

#[test]
fn remove_row_first() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(0), mat![[3, 4], [5, 6]]);
}

#[test]
fn remove_row_last() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(2), mat![[1, 2], [3, 4]]);
}

#[test]
fn remove_row_middle() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m.remove_row(1), mat![[1, 2], [5, 6]]);
}

#[test]
#[should_panic(expected = "Row index")]
fn remove_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.remove_row(2);
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
#[should_panic(expected = "Cannot reshape")]
fn reshape_mismatched_size() {
    let m = mat![[1, 2], [3, 4]];
    m.reshape(3, 3);
}

#[test]
fn reshape_same_shape() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.reshape(2, 2);
    assert_eq!(r, m);
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
fn resize_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.resize(2, 2, 0);
    assert_eq!(m, mat![[1, 2], [3, 4]]);
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
fn reverse_cols_double_is_identity() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let rr = m.reverse_cols().reverse_cols();
    assert_eq!(rr, m.as_ref());
}

#[test]
fn reverse_cols_empty() {
    let m: Mat<i32> = Mat::new();
    let r = m.reverse_cols();
    assert_eq!(r.shape(), (0, 0));
}

#[test]
fn reverse_rows_double_is_identity() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let rr = m.reverse_rows().reverse_rows();
    assert_eq!(rr, m.as_ref());
}

#[test]
fn reverse_rows_empty() {
    let m: Mat<i32> = Mat::new();
    let r = m.reverse_rows();
    assert_eq!(r.shape(), (0, 0));
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
#[should_panic(expected = "Column indices")]
fn swap_cols_out_of_bounds() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_cols(0, 2);
}

#[test]
fn swap_cols_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_cols(0, 0);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
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
#[should_panic(expected = "Row indices")]
fn swap_rows_out_of_bounds() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_rows(0, 2);
}

#[test]
fn swap_rows_same() {
    let mut m = mat![[1, 2], [3, 4]];
    m.swap_rows(0, 0);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(1, 0)], 3);
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
fn transpose_reverse_rows_is_reverse_cols_transpose() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let a = m.transpose().reverse_rows().to_owned();
    let b = m.reverse_cols().transpose().to_owned();
    assert_eq!(a, b);
}

#[test]
#[should_panic(expected = "Cannot truncate")]
fn truncate_larger_panics() {
    let mut m = mat![[1, 2], [3, 4]];
    m.truncate(3, 2);
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
fn vstack_basic() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6]];
    let m = Mat::vstack(&[a.as_ref(), b.as_ref()]);
    let expected = mat![[1, 2], [3, 4], [5, 6]];
    assert_eq!(m, expected);
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
fn vstack_single() {
    let a = mat![[1, 2], [3, 4]];
    let m = Mat::vstack(&[a.as_ref()]);
    assert_eq!(m, a);
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
