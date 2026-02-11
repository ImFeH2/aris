use aris::{Mat, mat};

#[test]
fn eq_element_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 0], [3, 5]];
    let result = a.eq_element(&b);
    assert_eq!(result, mat![[true, false], [true, false]]);
}

#[test]
fn eq_element_all_equal() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2], [3, 4]];
    let result = a.eq_element(&b);
    assert_eq!(result, mat![[true, true], [true, true]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn eq_element_shape_mismatch() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 2, 3]];
    a.eq_element(&b);
}

#[test]
fn ne_element_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[1, 0], [3, 5]];
    let result = a.ne_element(&b);
    assert_eq!(result, mat![[false, true], [false, true]]);
}

#[test]
fn lt_element_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[2, 2], [3, 5]];
    let result = a.lt_element(&b);
    assert_eq!(result, mat![[true, false], [false, true]]);
}

#[test]
fn le_element_2x2() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[2, 2], [3, 5]];
    let result = a.le_element(&b);
    assert_eq!(result, mat![[true, true], [true, true]]);
}

#[test]
fn gt_element_2x2() {
    let a = mat![[2, 2], [3, 5]];
    let b = mat![[1, 2], [3, 4]];
    let result = a.gt_element(&b);
    assert_eq!(result, mat![[true, false], [false, true]]);
}

#[test]
fn ge_element_2x2() {
    let a = mat![[2, 2], [3, 5]];
    let b = mat![[1, 2], [3, 4]];
    let result = a.ge_element(&b);
    assert_eq!(result, mat![[true, true], [true, true]]);
}

#[test]
fn logical_and_element_basic() {
    let a = mat![[true, false], [true, false]];
    let b = mat![[true, true], [false, false]];
    let result = a.logical_and_element(&b);
    assert_eq!(result, mat![[true, false], [false, false]]);
}

#[test]
fn logical_or_element_basic() {
    let a = mat![[true, false], [true, false]];
    let b = mat![[true, true], [false, false]];
    let result = a.logical_or_element(&b);
    assert_eq!(result, mat![[true, true], [true, false]]);
}

#[test]
fn logical_xor_element_basic() {
    let a = mat![[true, false], [true, false]];
    let b = mat![[true, true], [false, false]];
    let result = a.logical_xor_element(&b);
    assert_eq!(result, mat![[false, true], [true, false]]);
}

#[test]
fn logical_not_element_basic() {
    let a = mat![[true, false], [true, false]];
    let result = a.logical_not_element();
    assert_eq!(result, mat![[false, true], [false, true]]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn logical_and_element_shape_mismatch() {
    let a = mat![[true, false], [true, false]];
    let b = mat![[true, false, true]];
    a.logical_and_element(&b);
}

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

#[test]
fn combined_operations() {
    let a = mat![[1, 2], [3, 4]];
    let b = mat![[2, 2], [2, 2]];
    let lt = a.lt_element(&b);
    let gt = a.gt_element(&b);
    let result = lt.logical_or_element(&gt);
    assert_eq!(result, mat![[true, false], [true, true]]);
    assert!(!lt.all());
    assert!(lt.any());
}

#[test]
fn chained_logical_ops() {
    let a = mat![[true, false], [true, false]];
    let b = mat![[false, true], [false, true]];
    let c = mat![[true, true], [false, false]];
    let result = a.logical_or_element(&b).logical_and_element(&c);
    assert_eq!(result, mat![[true, true], [false, false]]);
}
