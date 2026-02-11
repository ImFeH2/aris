use aris::{Mat, mat};

#[test]
fn abs_f64() {
    let a = mat![[-1.5, 2.5], [3.0, -4.0]];
    let b = a.abs();
    assert_eq!(b, mat![[1.5, 2.5], [3.0, 4.0]]);
}

#[test]
fn abs_integers() {
    let a = mat![[1, -2], [-3, 4]];
    let b = a.abs();
    assert_eq!(b, mat![[1, 2], [3, 4]]);
}

#[test]
fn abs_on_matmut() {
    let mut a = mat![[-1, 2], [-3, 4]];
    let b = a.as_mut().abs();
    assert_eq!(b, mat![[1, 2], [3, 4]]);
}

#[test]
fn asin_acos_atan() {
    let a = mat![[0.0_f64, 0.5], [-0.5, 1.0]];
    let s = a.as_ref().asin();
    let c = a.as_ref().acos();
    for j in 0..2 {
        for i in 0..2 {
            assert!((s[(i, j)] + c[(i, j)] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }
    }
    let t = a.as_ref().atan();
    assert!((t[(0, 0)]).abs() < 1e-10);
}

#[test]
fn asinh_acosh_atanh() {
    let val = 2.0_f64;
    let a = mat![[val]];
    let b = a.as_ref().sinh().as_ref().asinh();
    assert!((b[(0, 0)] - val).abs() < 1e-10);

    let a2 = mat![[val]];
    let b2 = a2.as_ref().cosh().as_ref().acosh();
    assert!((b2[(0, 0)] - val).abs() < 1e-10);

    let a3 = mat![[0.5_f64]];
    let b3 = a3.as_ref().tanh().as_ref().atanh();
    assert!((b3[(0, 0)] - 0.5).abs() < 1e-10);
}

#[test]
fn cbrt_basic() {
    let a = mat![[8.0, 27.0], [64.0, 125.0]];
    let b = a.cbrt();
    assert_eq!(b, mat![[2.0, 3.0], [4.0, 5.0]]);
}

#[test]
fn ceil_floor_round() {
    let a = mat![[1.2, 2.5], [3.7, -1.3]];
    let c = a.ceil();
    assert_eq!(c, mat![[2.0, 3.0], [4.0, -1.0]]);
    let f = a.floor();
    assert_eq!(f, mat![[1.0, 2.0], [3.0, -2.0]]);
    let r = a.round();
    assert_eq!(r, mat![[1.0, 3.0], [4.0, -1.0]]);
}

#[test]
fn clamp_all_within() {
    let a = mat![[3, 4], [5, 6]];
    let b = a.clamp(1, 10);
    assert_eq!(b, a);
}

#[test]
fn clamp_basic() {
    let a = mat![[1, 5, 3], [8, 2, 7]];
    let b = a.clamp(2, 6);
    assert_eq!(b, mat![[2, 5, 3], [6, 2, 6]]);
}

#[test]
fn clamp_f64() {
    let a = mat![[-1.0, 0.5], [1.5, 2.0]];
    let b = a.clamp(0.0, 1.0);
    assert_eq!(b, mat![[0.0, 0.5], [1.0, 1.0]]);
}

#[test]
fn exp_ln_roundtrip() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = a.exp().as_ref().ln();
    for j in 0..2 {
        for i in 0..2 {
            assert!((b[(i, j)] - a[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn log10_basic() {
    let a = mat![[1.0, 10.0], [100.0, 1000.0]];
    let b = a.log10();
    assert_eq!(b, mat![[0.0, 1.0], [2.0, 3.0]]);
}

#[test]
fn log2_basic() {
    let a = mat![[1.0, 2.0], [4.0, 8.0]];
    let b = a.log2();
    assert_eq!(b, mat![[0.0, 1.0], [2.0, 3.0]]);
}

#[test]
fn map_double() {
    let a = mat![[1, 2], [3, 4]];
    let b = a.map(|x| x * 2);
    assert_eq!(b, mat![[2, 4], [6, 8]]);
}

#[test]
fn map_empty() {
    let a: Mat<i32> = Mat::zeros(0, 0);
    let b = a.map(|x| x + 1);
    assert_eq!(b.shape(), (0, 0));
}

#[test]
fn map_on_matmut() {
    let mut a = mat![[1, 2], [3, 4]];
    let b = a.as_mut().map(|x| x * 3);
    assert_eq!(b, mat![[3, 6], [9, 12]]);
}

#[test]
fn map_on_transpose() {
    let a = mat![[1, 2], [3, 4]];
    let b = a.transpose().map(|x| x * 10);
    assert_eq!(b, mat![[10, 30], [20, 40]]);
}

#[test]
fn map_type_change() {
    let a = mat![[1, 2], [3, 4]];
    let b: Mat<f64> = a.map(|x| *x as f64 + 0.5);
    assert_eq!(b, mat![[1.5, 2.5], [3.5, 4.5]]);
}

#[test]
fn pow_basic() {
    let a = mat![[1.0, 2.0], [3.0, 4.0]];
    let b = a.pow(2.0);
    assert_eq!(b, mat![[1.0, 4.0], [9.0, 16.0]]);
}

#[test]
fn pow_sqrt_via_half() {
    let a = mat![[4.0, 9.0], [16.0, 25.0]];
    let b = a.pow(0.5);
    assert_eq!(b, mat![[2.0, 3.0], [4.0, 5.0]]);
}

#[test]
fn signum_f64() {
    let a = mat![[-2.5, 0.0], [3.0, -0.0]];
    let b = a.signum();
    assert_eq!(b[(0, 0)], -1.0);
    assert_eq!(b[(1, 0)], 1.0);
}

#[test]
fn signum_integers() {
    let a = mat![[5, -3], [0, 7]];
    let b = a.signum();
    assert_eq!(b, mat![[1, -1], [0, 1]]);
}

#[test]
fn sin_cos_identity() {
    let a = mat![
        [0.0_f64, std::f64::consts::FRAC_PI_6],
        [std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2]
    ];
    let s = a.sin();
    let c = a.cos();
    for j in 0..2 {
        for i in 0..2 {
            let sum = s[(i, j)] * s[(i, j)] + c[(i, j)] * c[(i, j)];
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn sinh_cosh_identity() {
    let a = mat![[0.5_f64, 1.0], [1.5, 2.0]];
    let sh = a.sinh();
    let ch = a.cosh();
    for j in 0..2 {
        for i in 0..2 {
            let diff = ch[(i, j)] * ch[(i, j)] - sh[(i, j)] * sh[(i, j)];
            assert!((diff - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn sqrt_basic() {
    let a = mat![[1.0, 4.0], [9.0, 16.0]];
    let b = a.sqrt();
    assert_eq!(b, mat![[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn tan_basic() {
    let a = mat![[0.0_f64]];
    let b = a.tan();
    assert!((b[(0, 0)]).abs() < 1e-10);
}

#[test]
fn tanh_range() {
    let a = mat![[-10.0_f64, 0.0], [0.5, 10.0]];
    let b = a.tanh();
    for j in 0..2 {
        for i in 0..2 {
            assert!(b[(i, j)] >= -1.0 && b[(i, j)] <= 1.0);
        }
    }
    assert!((b[(0, 1)]).abs() < 1e-10);
}

#[test]
fn zip_map_add() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[10, 20], [30, 40]];
    let c = a.zip_map(b.as_ref(), |x, y| x + y);
    assert_eq!(c, mat![[11, 22], [33, 44]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn zip_map_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3]];
    let _ = a.zip_map(b.as_ref(), |x, y| x + y);
}

#[test]
fn zip_map_type_change() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2], [3, 5]];
    let c: Mat<bool> = a.zip_map(b.as_ref(), |x, y| x == y);
    assert_eq!(c, mat![[true, true], [true, false]]);
}
