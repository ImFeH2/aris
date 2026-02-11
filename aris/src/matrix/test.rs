use super::*;

#[test]
fn ptr_at_contiguous() {
    let m = Mat::from_vec_col(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let mr = m.as_ref();

    let ptr0 = mr.ptr_at(0, 0);
    let ptr1 = mr.ptr_at(1, 0);
    let ptr2 = mr.ptr_at(0, 1);

    unsafe {
        assert_eq!(*ptr0, 1);
        assert_eq!(*ptr1, 2);
        assert_eq!(*ptr2, 3);

        assert_eq!(ptr1.offset_from(ptr0), 1);
        assert_eq!(ptr2.offset_from(ptr0), 2);
    }
}

#[test]
fn ptr_at_transpose() {
    let m = Mat::from_vec_col(3, 4, (0..12).collect());
    let mt = m.transpose();

    let ptr0 = mt.ptr_at(0, 0);
    let ptr1 = mt.ptr_at(0, 1);

    unsafe {
        assert_eq!(*ptr0, 0);
        assert_eq!(*ptr1, 1);
        assert_eq!(ptr1.offset_from(ptr0), 1);
    }
}

#[test]
fn at_consistency() {
    let m = Mat::from_fn(4, 5, |i, j| i * 10 + j);
    let mr = m.as_ref();

    for j in 0..5 {
        for i in 0..4 {
            assert_eq!(*mr.at(i, j), m[(i, j)]);
        }
    }
}

#[test]
fn row_stride_contiguous() {
    let m = Mat::from_vec_col(3, 4, (0..12).collect());
    let mr = m.as_ref();

    assert_eq!(mr.row_stride(), 1);
}

#[test]
fn col_stride_contiguous() {
    let m = Mat::from_vec_col(3, 4, (0..12).collect());
    let mr = m.as_ref();

    assert_eq!(mr.col_stride(), 3);
}

#[test]
fn strides_transpose() {
    let m = Mat::from_vec_col(3, 4, (0..12).collect());
    let mt = m.transpose();

    assert_eq!(mt.row_stride(), 3);
    assert_eq!(mt.col_stride(), 1);
}

#[test]
fn strides_view() {
    let m = Mat::from_vec_col(5, 6, (0..30).collect());
    let sub = m.view(1, 2, 3, 2);

    assert_eq!(sub.row_stride(), 1);
    assert_eq!(sub.col_stride(), 5);
}

#[test]
fn strides_row_view() {
    let m = Mat::from_vec_col(4, 5, (0..20).collect());
    let row = m.row(2);

    assert_eq!(row.row_stride(), 1);
    assert_eq!(row.col_stride(), 4);
}

#[test]
fn strides_col_view() {
    let m = Mat::from_vec_col(4, 5, (0..20).collect());
    let col = m.col(2);

    assert_eq!(col.row_stride(), 1);
    assert_eq!(col.col_stride(), 4);
}
