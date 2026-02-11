use aris::Mat;

#[test]
fn col_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.col_iter().count(), 0);
}

#[test]
fn col_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.col_iter();
    assert_eq!(iter.len(), 3);
}

#[test]
fn col_iter_mut_exact_size() {
    let mut m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.col_iter_mut();
    assert_eq!(iter.len(), 3);
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
fn enumerate_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.enumerate().count(), 0);
}

#[test]
fn enumerate_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.enumerate();
    assert_eq!(iter.len(), 6);
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
fn row_iter_empty_matrix() {
    let m: Mat<i32> = Mat::new();
    assert_eq!(m.row_iter().count(), 0);
}

#[test]
fn row_iter_exact_size() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.row_iter();
    assert_eq!(iter.len(), 2);
}

#[test]
fn row_iter_mut_exact_size() {
    let mut m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let iter = m.row_iter_mut();
    assert_eq!(iter.len(), 2);
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
