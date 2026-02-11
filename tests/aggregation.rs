use aris::{Mat, mat};

#[test]
fn sum_basic() {
    let a = mat![[1, 2], [3, 4]];
    assert_eq!(a.sum(), 10);
}

#[test]
fn sum_single_element() {
    let a = mat![[5]];
    assert_eq!(a.sum(), 5);
}

#[test]
fn sum_empty() {
    let a: Mat<i32> = Mat::new();
    assert_eq!(a.sum(), 0);
}

#[test]
fn prod_basic() {
    let a = mat![[1, 2], [3, 4]];
    assert_eq!(a.prod(), 24);
}

#[test]
fn prod_with_zero() {
    let a = mat![[1, 0], [3, 4]];
    assert_eq!(a.prod(), 0);
}

#[test]
fn mean_basic() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(a.mean(), 2.5);
}

#[test]
fn mean_single() {
    let a = mat![[5.0]];
    assert_eq!(a.mean(), 5.0);
}

#[test]
#[should_panic(expected = "cannot compute mean of empty matrix")]
fn mean_empty_panics() {
    let a: Mat<f64> = Mat::new();
    let _ = a.mean();
}

#[test]
fn variance_basic() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let var = a.variance();
    assert!((var - 1.25_f64).abs() < 1e-10);
}

#[test]
fn std_dev_basic() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let std = a.std_dev();
    let expected = 1.25_f64.sqrt();
    assert!((std - expected).abs() < 1e-10);
}

#[test]
fn min_basic() {
    let a = mat![[3, 1], [4, 2]];
    assert_eq!(a.min(), 1);
}

#[test]
fn min_negative() {
    let a = mat![[-5, 3], [1, -2]];
    assert_eq!(a.min(), -5);
}

#[test]
#[should_panic(expected = "cannot compute min of empty matrix")]
fn min_empty_panics() {
    let a: Mat<i32> = Mat::new();
    let _ = a.min();
}

#[test]
fn max_basic() {
    let a = mat![[3, 1], [4, 2]];
    assert_eq!(a.max(), 4);
}

#[test]
fn max_negative() {
    let a = mat![[-5, -3], [-1, -2]];
    assert_eq!(a.max(), -1);
}

#[test]
fn min_max_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.min_max(), (0, 5));
}

#[test]
fn sum_rows_basic() {
    let a = mat![[1, 2, 3], [4, 5, 6]];
    let result = a.sum_rows();
    assert_eq!(result, mat![[6], [15]]);
}

#[test]
fn sum_cols_basic() {
    let a = mat![[1, 2, 3], [4, 5, 6]];
    let result = a.sum_cols();
    assert_eq!(result, mat![[5, 7, 9]]);
}

#[test]
fn mean_rows_basic() {
    let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = a.mean_rows();
    assert_eq!(result, mat![[2.0], [5.0]]);
}

#[test]
fn mean_cols_basic() {
    let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = a.mean_cols();
    assert_eq!(result, mat![[2.5, 3.5, 4.5]]);
}

#[test]
fn min_rows_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    let result = a.min_rows();
    assert_eq!(result, mat![[1], [0]]);
}

#[test]
fn min_cols_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    let result = a.min_cols();
    assert_eq!(result, mat![[3, 1, 0]]);
}

#[test]
fn max_rows_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    let result = a.max_rows();
    assert_eq!(result, mat![[5], [4]]);
}

#[test]
fn max_cols_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    let result = a.max_cols();
    assert_eq!(result, mat![[4, 2, 5]]);
}

#[test]
fn argmin_basic() {
    let a = mat![[3, 1], [4, 2]];
    assert_eq!(a.argmin(), (0, 1));
}

#[test]
fn argmin_first_occurrence() {
    let a = mat![[1, 3], [1, 4]];
    assert_eq!(a.argmin(), (0, 0));
}

#[test]
fn argmax_basic() {
    let a = mat![[3, 1], [4, 2]];
    assert_eq!(a.argmax(), (1, 0));
}

#[test]
fn argmax_last_column() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.argmax(), (0, 2));
}

#[test]
fn argmin_col_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.argmin_col(), vec![0, 0, 1]);
}

#[test]
fn argmax_col_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.argmax_col(), vec![1, 1, 0]);
}

#[test]
fn argmin_row_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.argmin_row(), vec![1, 2]);
}

#[test]
fn argmax_row_basic() {
    let a = mat![[3, 1, 5], [4, 2, 0]];
    assert_eq!(a.argmax_row(), vec![2, 0]);
}

#[test]
fn cumsum_col_basic() {
    let a = mat![[1, 2], [3, 4], [5, 6]];
    let result = a.cumsum_col();
    assert_eq!(result, mat![[1, 2], [4, 6], [9, 12]]);
}

#[test]
fn cumsum_row_basic() {
    let a = mat![[1, 2, 3], [4, 5, 6]];
    let result = a.cumsum_row();
    assert_eq!(result, mat![[1, 3, 6], [4, 9, 15]]);
}

#[test]
fn cumsum_col_single_col() {
    let a = mat![[1], [2], [3]];
    let result = a.cumsum_col();
    assert_eq!(result, mat![[1], [3], [6]]);
}

#[test]
fn cumprod_col_basic() {
    let a = mat![[1, 2], [3, 4], [5, 6]];
    let result = a.cumprod_col();
    assert_eq!(result, mat![[1, 2], [3, 8], [15, 48]]);
}

#[test]
fn cumprod_row_basic() {
    let a = mat![[1, 2, 3], [4, 5, 6]];
    let result = a.cumprod_row();
    assert_eq!(result, mat![[1, 2, 6], [4, 20, 120]]);
}

#[test]
fn cumprod_with_zero() {
    let a = mat![[1, 2], [0, 3], [4, 5]];
    let result = a.cumprod_col();
    assert_eq!(result, mat![[1, 2], [0, 6], [0, 30]]);
}

#[test]
fn sort_rows_by_col_first() {
    let a = mat![[3, 1], [1, 2], [2, 3]];
    let result = a.sort_rows_by_col(0);
    assert_eq!(result, mat![[1, 2], [2, 3], [3, 1]]);
}

#[test]
fn sort_rows_by_col_second() {
    let a = mat![[3, 1], [1, 2], [2, 3]];
    let result = a.sort_rows_by_col(1);
    assert_eq!(result, mat![[3, 1], [1, 2], [2, 3]]);
}

#[test]
fn sort_cols_by_row_first() {
    let a = mat![[3, 1, 2], [1, 2, 3]];
    let result = a.sort_cols_by_row(0);
    assert_eq!(result, mat![[1, 2, 3], [2, 3, 1]]);
}

#[test]
fn sort_cols_by_row_second() {
    let a = mat![[3, 1, 2], [1, 2, 3]];
    let result = a.sort_cols_by_row(1);
    assert_eq!(result, mat![[3, 1, 2], [1, 2, 3]]);
}

#[test]
#[should_panic(expected = "column index 2 out of bounds")]
fn sort_rows_by_col_out_of_bounds() {
    let a = mat![[1, 2], [3, 4]];
    let _ = a.sort_rows_by_col(2);
}

#[test]
#[should_panic(expected = "row index 2 out of bounds")]
fn sort_cols_by_row_out_of_bounds() {
    let a = mat![[1, 2], [3, 4]];
    let _ = a.sort_cols_by_row(2);
}

#[test]
fn sum_float() {
    let a = mat![[1.5, 2.5], [3.5, 4.5]];
    assert_eq!(a.sum(), 12.0);
}

#[test]
fn prod_float() {
    let a = mat![[2.0, 3.0], [4.0, 5.0]];
    assert_eq!(a.prod(), 120.0);
}

#[test]
fn cumsum_col_empty() {
    let a: Mat<i32> = Mat::new();
    let result = a.cumsum_col();
    assert_eq!(result.shape(), (0, 0));
}

#[test]
fn cumsum_row_empty() {
    let a: Mat<i32> = Mat::new();
    let result = a.cumsum_row();
    assert_eq!(result.shape(), (0, 0));
}
