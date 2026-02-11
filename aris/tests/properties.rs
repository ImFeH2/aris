use aris::{Mat, col, mat, row};

#[test]
fn is_col_vector_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_col_vector());
}

#[test]
fn is_col_vector_true() {
    let c = col![1, 2, 3];
    assert!(c.is_col_vector());
}

#[test]
fn is_diagonal_false() {
    let m = mat![[1, 2], [0, 3]];
    assert!(!m.is_diagonal());
}

#[test]
fn is_diagonal_identity() {
    let m: Mat<i32> = Mat::identity(3);
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
fn is_diagonal_true() {
    let m = Mat::diag(&[1, 2, 3]);
    assert!(m.is_diagonal());
}

#[test]
fn is_diagonal_zeros() {
    let m: Mat<i32> = Mat::zeros(3, 3);
    assert!(m.is_diagonal());
}

#[test]
fn is_empty_false_for_nonempty() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_empty());
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
fn is_identity_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_identity());
}

#[test]
fn is_identity_1x1() {
    let m = mat![[1]];
    assert!(m.is_identity());
}

#[test]
fn is_identity_false_diagonal() {
    let m = mat![[1, 0], [0, 2]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_false_off_diagonal() {
    let m = mat![[1, 1], [0, 1]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_non_square() {
    let m = mat![[1, 0, 0], [0, 1, 0]];
    assert!(!m.is_identity());
}

#[test]
fn is_identity_true() {
    let m: Mat<i32> = Mat::identity(4);
    assert!(m.is_identity());
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
fn is_lower_triangular_true() {
    let m = mat![[1, 0, 0], [2, 3, 0], [4, 5, 6]];
    assert!(m.is_lower_triangular());
}

#[test]
fn is_row_vector_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_row_vector());
}

#[test]
fn is_row_vector_true() {
    let r = row![1, 2, 3];
    assert!(r.is_row_vector());
}

#[test]
fn is_scalar_false_col_vec() {
    let m = col![1, 2];
    assert!(!m.is_scalar());
}

#[test]
fn is_scalar_false_row_vec() {
    let m = row![1, 2];
    assert!(!m.is_scalar());
}

#[test]
fn is_scalar_true() {
    let m = mat![[42]];
    assert!(m.is_scalar());
}

#[test]
fn is_square_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_square());
}

#[test]
fn is_square_false() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert!(!m.is_square());
}

#[test]
fn is_square_true() {
    let m = mat![[1, 2], [3, 4]];
    assert!(m.is_square());
}

#[test]
fn is_symmetric_0x0() {
    let m: Mat<i32> = Mat::new();
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_1x1() {
    let m = mat![[7]];
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_false() {
    let m = mat![[1, 2], [3, 4]];
    assert!(!m.is_symmetric());
}

#[test]
fn is_symmetric_identity() {
    let m: Mat<i32> = Mat::identity(3);
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_non_square() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    assert!(!m.is_symmetric());
}

#[test]
fn is_symmetric_true() {
    let m = mat![[1, 2, 3], [2, 5, 6], [3, 6, 9]];
    assert!(m.is_symmetric());
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
fn is_upper_triangular_true() {
    let m = mat![[1, 2, 3], [0, 4, 5], [0, 0, 6]];
    assert!(m.is_upper_triangular());
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
fn shape_predicates_on_mat_ref() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.as_ref();
    assert!(!r.is_empty());
    assert!(r.is_square());
    assert!(!r.is_row_vector());
    assert!(!r.is_col_vector());
    assert!(!r.is_scalar());
}
