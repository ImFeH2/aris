mod common;

use aris::{Mat, mat};

use common::c;

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
fn add_assign_mat_ref_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    a += &b;
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
#[should_panic(expected = "shape mismatch")]
fn add_assign_shape_mismatch() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3]];
    a += &b;
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
fn add_empty_matrices() {
    let a: Mat<i32> = Mat::zeros(0, 0);
    let b: Mat<i32> = Mat::zeros(0, 0);
    let c = &a + &b;
    assert_eq!(c.shape(), (0, 0));
}

#[test]
fn add_mat_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a + b;
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
fn add_mat_matref() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a + b.as_ref();
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
fn add_matmut_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_mut() + b;
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
fn add_matmut_ref_mat() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_mut() + &b;
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
fn add_matref_matmut() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let c = a.as_ref() + b.as_mut();
    assert_eq!(c, mat![[6, 8], [10, 12]]);
}

#[test]
fn add_matref_matref() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = a.as_ref() + b.as_ref();
    assert_eq!(c, mat![[11, 22], [33, 44]]);
}

#[test]
fn add_matref_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = a.as_ref() + &b;
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
fn add_ref_mat_matmut() {
    let a = mat![[1, 2], [3, 4]];
    let mut b = mat![[5, 6], [7, 8]];
    let c = &a + b.as_mut();
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
fn add_ref_mat_ref_mat() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = &a + &b;
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
fn arithmetic_chain() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = &a + &b - &a;
    assert_eq!(c, b);
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
fn mul_scalar_f64() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let c = 2.5 * &a;
    assert_eq!(c, mat![[2.5, 5.0], [7.5, 10.0]]);
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
fn neg_ref_matmut() {
    let mut a = mat![[1, -2], [3, -4]];
    let am = a.as_mut();
    let c = -&am;
    assert_eq!(c, mat![[-1, 2], [-3, 4]]);
}

#[test]
fn neg_then_add() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[5, 6], [7, 8]];
    let c = -&a + &b;
    assert_eq!(c, mat![[4, 4], [4, 4]]);
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
fn scalar_ops_chain() {
    let a = mat![[2, 4], [6, 8]];
    let c = &a * 3 / 2;
    assert_eq!(c, mat![[3, 6], [9, 12]]);
}

#[test]
fn sub_assign_mat_mat() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    a -= b;
    assert_eq!(a, mat![[9, 18], [27, 36]]);
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
fn sub_mat_mat() {
    let a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = a - b;
    assert_eq!(c, mat![[9, 18], [27, 36]]);
}

#[test]
fn sub_matmut_matref() {
    let mut a = mat![[10, 20], [30, 40]];
    let b = mat![[1, 2], [3, 4]];
    let c = a.as_mut() - b.as_ref();
    assert_eq!(c, mat![[9, 18], [27, 36]]);
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
#[should_panic(expected = "shape mismatch")]
fn sub_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1], [2]];
    let _ = &a - &b;
}
