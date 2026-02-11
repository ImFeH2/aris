mod common;

use aris::{Mat, mat};

use common::c;

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
fn as_ref_creates_view() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 2);
    assert_eq!(r[(0, 0)], 1);
    assert_eq!(r[(1, 1)], 4);
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
fn col_view() {
    let m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let c = m.col(2);
    assert_eq!(c.shape(), (3, 1));
    assert_eq!(c[(0, 0)], 3);
    assert_eq!(c[(1, 0)], 6);
    assert_eq!(c[(2, 0)], 9);
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
#[should_panic(expected = "Column index")]
fn col_view_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.col(2);
}

#[test]
fn cols_range_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.cols_range(0..0);
    assert_eq!(sub.shape(), (2, 0));
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
fn diagonal_of_transpose() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let d = m.transpose().diagonal();
    assert_eq!(d.shape(), (2, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 5);
}

#[test]
fn diagonal_view_empty() {
    let m: Mat<i32> = Mat::new();
    let d = m.diagonal();
    assert_eq!(d.shape(), (0, 1));
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
fn diagonal_view_tall() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let d = m.diagonal();
    assert_eq!(d.shape(), (2, 1));
    assert_eq!(d[(0, 0)], 1);
    assert_eq!(d[(1, 0)], 4);
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
fn mat_mut_debug() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    assert_eq!(format!("{:?}", v), "MatMut(2x2, [[1, 2], [3, 4]])");
}

#[test]
fn mat_mut_diag_iter() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    let diag: Vec<_> = v.diag_iter().collect();
    assert_eq!(diag, vec![&1, &4]);
}

#[test]
fn mat_mut_display() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let v = m.as_mut();
    assert_eq!(format!("{}", v), "[[1, 2],\n [3, 4]]");
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
fn mat_mut_flatten() {
    let mut m = mat![[1, 2], [3, 4]];
    let v = m.as_mut();
    let f = v.flatten();
    assert_eq!(f.shape(), (4, 1));
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
fn mat_mut_index_panics() {
    let mut m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let v = m.as_mut();
        let _ = v[(2, 0)];
    }));
    assert!(result.is_err());
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
fn mat_mut_reshape() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    let v = m.as_mut();
    let r = v.reshape(3, 2);
    assert_eq!(r.shape(), (3, 2));
}

#[test]
fn mat_mut_row_col_readonly() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    let v = m.as_mut();
    assert_eq!(v.row(0)[(0, 1)], 2);
    assert_eq!(v.col(2)[(1, 0)], 6);
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
fn mat_mut_submatrix_readonly() {
    let mut m = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let v = m.as_mut();
    let sub = v.submatrix(1, 0, 2, 2);
    assert_eq!(sub[(0, 0)], 4);
    assert_eq!(sub[(1, 1)], 8);
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
fn mat_ref_debug() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(format!("{:?}", r), "MatRef(2x2, [[1, 2], [3, 4]])");
}

#[test]
fn mat_ref_display() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    assert_eq!(format!("{}", r), "[[1, 2],\n [3, 4]]");
}

#[test]
fn mat_ref_get() {
    let m = Mat::from_rows(&[&[1, 2, 3], &[4, 5, 6]]);
    let r = m.as_ref();
    assert_eq!(r.get(0, 2), Some(&3));
    assert_eq!(r.get(2, 0), None);
}

#[test]
fn mat_ref_index_panics() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let result = std::panic::catch_unwind(|| r[(2, 0)]);
    assert!(result.is_err());
}

#[test]
fn mat_ref_insert_row() {
    let m = mat![[1, 2], [3, 4]];
    let r = m.as_ref().insert_row(1, &[5, 6]);
    assert_eq!(r, mat![[1, 2], [5, 6], [3, 4]]);
}

#[test]
fn mat_ref_is_copy() {
    let m = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let r = m.as_ref();
    let r2 = r;
    assert_eq!(r[(0, 0)], r2[(0, 0)]);
}

#[test]
fn mat_ref_partial_eq() {
    let a = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    let b = Mat::from_rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(a.as_ref(), b.as_ref());
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
fn math_on_view() {
    let a = mat![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]];
    let sub = a.submatrix(0, 0, 2, 2);
    let b = sub.sqrt();
    assert_eq!(b, mat![[1.0, 2.0], [4.0, 5.0]]);
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
fn reverse_cols_mut_view() {
    let mut m = mat![[1, 2, 3], [4, 5, 6]];
    {
        let mut r = m.reverse_cols_mut();
        r[(0, 0)] = 30;
    }
    assert_eq!(m[(0, 2)], 30);
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
fn reverse_rows_mut_view() {
    let mut m = mat![[1, 2], [3, 4], [5, 6]];
    {
        let mut r = m.reverse_rows_mut();
        r[(0, 0)] = 50;
    }
    assert_eq!(m[(2, 0)], 50);
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
fn reversed_view_to_owned() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let owned = m.reverse_rows().to_owned();
    assert_eq!(owned, mat![[5, 6], [3, 4], [1, 2]]);
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
fn row_of_reversed_matrix() {
    let m = mat![[1, 2], [3, 4], [5, 6]];
    let r = m.reverse_rows().row(0);
    assert_eq!(r[(0, 0)], 5);
    assert_eq!(r[(0, 1)], 6);
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
#[should_panic(expected = "Row index")]
fn row_view_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.row(2);
}

#[test]
fn rows_range_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.rows_range(1..1);
    assert_eq!(sub.shape(), (0, 2));
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
#[should_panic(expected = "Split index")]
fn split_at_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.split_at_col(3);
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
#[should_panic(expected = "Split index")]
fn split_at_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.split_at_row(3);
}

#[test]
#[should_panic(expected = "Column range")]
fn submatrix_col_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.submatrix(0, 1, 1, 2);
}

#[test]
fn submatrix_empty() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.submatrix(0, 0, 0, 0);
    assert_eq!(sub.shape(), (0, 0));
}

#[test]
fn submatrix_full() {
    let m = mat![[1, 2], [3, 4]];
    let sub = m.submatrix(0, 0, 2, 2);
    assert_eq!(sub, m.as_ref());
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
#[should_panic(expected = "Row range")]
fn submatrix_row_out_of_bounds() {
    let m = mat![[1, 2], [3, 4]];
    m.submatrix(1, 0, 2, 1);
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
fn transpose_mut_view() {
    let mut m = mat![[1, 2], [3, 4]];
    {
        let mut t = m.transpose_mut();
        t[(0, 1)] = 30;
    }
    assert_eq!(m[(1, 0)], 30);
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
fn transposed_view_to_owned() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let owned = m.transpose().to_owned();
    assert_eq!(owned, mat![[1, 4], [2, 5], [3, 6]]);
}

#[test]
fn view_to_owned() {
    let m = mat![[1, 2, 3], [4, 5, 6]];
    let sub = m.submatrix(0, 1, 2, 2);
    let owned = sub.to_owned();
    assert_eq!(owned, mat![[2, 3], [5, 6]]);
}
