mod common;

use aris::{Complex, Mat, mat};

use common::c;

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
fn complex_arg() {
    let m = mat![[c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0), c(0.0, -1.0)]];
    let a = m.arg();
    assert!((a[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((a[(0, 1)] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((a[(0, 2)] - std::f64::consts::PI).abs() < 1e-10);
    assert!((a[(0, 3)] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
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
fn complex_construct_and_equality() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m[(0, 0)], c(1.0, 2.0));
    assert_eq!(m[(1, 1)], c(7.0, 8.0));
}

#[test]
fn complex_div_assign() {
    let mut a = mat![[c(4.0, 2.0)]];
    a /= c(2.0, 0.0);
    assert_eq!(a, mat![[c(2.0, 1.0)]]);
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
fn complex_is_hermitian_1x1() {
    let m = mat![[c(3.0, 0.0)]];
    assert!(m.is_hermitian());
    let m2 = mat![[c(3.0, 1.0)]];
    assert!(!m2.is_hermitian());
}

#[test]
fn complex_is_hermitian_false_diag() {
    let m = mat![[c(1.0, 1.0), c(2.0, 3.0)], [c(2.0, -3.0), c(5.0, 0.0)]];
    assert!(!m.is_hermitian());
}

#[test]
fn complex_is_hermitian_false_offdiag() {
    let m = mat![[c(1.0, 0.0), c(2.0, 3.0)], [c(2.0, 3.0), c(5.0, 0.0)]];
    assert!(!m.is_hermitian());
}

#[test]
fn complex_is_hermitian_nonsquare() {
    let m = mat![[c(1.0, 0.0), c(2.0, 0.0)]];
    assert!(!m.is_hermitian());
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
fn complex_is_symmetric() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(3.0, 4.0), c(5.0, 6.0)]];
    assert!(m.is_symmetric());
}

#[test]
fn complex_log() {
    let m = mat![[c(1.0, 0.0)]];
    let result = m.log(std::f64::consts::E);
    assert!((result[(0, 0)] - c(0.0, 0.0)).norm() < 1e-10);
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
fn complex_map_sqrt() {
    let m = mat![[c(0.0, 0.0), c(4.0, 0.0), c(-1.0, 0.0)]];
    let result = m.map(|x| x.sqrt());
    assert!((result[(0, 0)] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 1)] - c(2.0, 0.0)).norm() < 1e-10);
    assert!((result[(0, 2)] - c(0.0, 1.0)).norm() < 1e-10);
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
fn complex_neg() {
    let a = mat![[c(1.0, 2.0), c(-3.0, 4.0)]];
    let result = -&a;
    assert_eq!(result, mat![[c(-1.0, -2.0), c(3.0, -4.0)]]);
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
fn complex_re_im() {
    let m = mat![[c(1.0, 2.0), c(3.0, 4.0)], [c(5.0, 6.0), c(7.0, 8.0)]];
    let re = m.re();
    let im = m.im();
    assert_eq!(re, mat![[1.0, 3.0], [5.0, 7.0]]);
    assert_eq!(im, mat![[2.0, 4.0], [6.0, 8.0]]);
}

#[test]
fn complex_re_im_on_mut() {
    let mut m = mat![[c(1.0, 2.0)]];
    let mm = m.as_mut();
    assert_eq!(mm.re(), mat![[1.0]]);
    assert_eq!(mm.im(), mat![[2.0]]);
}

#[test]
fn complex_re_im_on_ref() {
    let m = mat![[c(1.0, 2.0), c(3.0, -1.0)]];
    let r = m.as_ref();
    assert_eq!(r.re(), mat![[1.0, 3.0]]);
    assert_eq!(r.im(), mat![[2.0, -1.0]]);
}

#[test]
fn complex_scalar_div() {
    let a = mat![[c(4.0, 2.0), c(6.0, 0.0)]];
    let result = &a / c(2.0, 0.0);
    assert_eq!(result, mat![[c(2.0, 1.0), c(3.0, 0.0)]]);
}

#[test]
fn complex_scalar_lmul() {
    let a = mat![[c(1.0, 2.0), c(3.0, 4.0)]];
    let result = c(2.0, 0.0) * &a;
    assert_eq!(result, mat![[c(2.0, 4.0), c(6.0, 8.0)]]);
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
