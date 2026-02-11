use aris::{Mat, mat};

#[test]
fn all_true() {
    let a = mat![[true, true], [true, true]];
    assert!(a.all());
}

#[test]
fn all_false_with_one_false() {
    let a = mat![[true, true], [true, false]];
    assert!(!a.all());
}

#[test]
fn all_empty() {
    let a: Mat<bool> = Mat::new();
    assert!(a.all());
}

#[test]
fn any_true() {
    let a = mat![[false, false], [false, true]];
    assert!(a.any());
}

#[test]
fn any_false_all_false() {
    let a = mat![[false, false], [false, false]];
    assert!(!a.any());
}

#[test]
fn any_empty() {
    let a: Mat<bool> = Mat::new();
    assert!(!a.any());
}
