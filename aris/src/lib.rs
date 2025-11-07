mod matrix;

#[cfg(test)]
mod tests {
    use crate::{mat, matrix::Matrix};

    #[test]
    fn test_zeros() {
        let m: Matrix<i32> = Matrix::zeros((2, 3));
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(1, 2)], 0);
    }

    #[test]
    fn test_from_vec() {
        let m = Matrix::from_vec((2, 3), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    #[should_panic]
    fn test_from_vec_wrong_size() {
        let _m = Matrix::from_vec((2, 3), vec![1, 2, 3]);
    }

    #[test]
    fn test_from_nested_vec() {
        let m = Matrix::from_nested_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 1)], 2);
        assert_eq!(m[(1, 0)], 4);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Row 1 has 2 elements, expected 3")]
    fn test_from_nested_vec_inconsistent() {
        let _m = Matrix::from_nested_vec(vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn test_from_value() {
        let m = Matrix::from_value((3, 2), 42);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);
        assert_eq!(m[(0, 0)], 42);
        assert_eq!(m[(2, 1)], 42);
    }

    #[test]
    fn test_from_fn() {
        let m = Matrix::from_fn((3, 3), |i, j| i * 3 + j);
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(0, 1)], 1);
        assert_eq!(m[(1, 0)], 3);
        assert_eq!(m[(2, 2)], 8);
    }

    #[test]
    fn test_get() {
        let m = mat![[1, 2], [3, 4]];
        assert_eq!(m.get((0, 0)), Some(&1));
        assert_eq!(m.get((1, 1)), Some(&4));
        assert_eq!(m.get((1, 2)), None);
        assert_eq!(m.get((2, 1)), None);
    }

    #[test]
    fn test_index() {
        let m = mat![[1, 2, 3], [4, 5, 6]];
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 2)], 3);
        assert_eq!(m[(1, 1)], 5);
    }

    #[test]
    fn test_index_mut() {
        let mut m = mat![[1, 2], [3, 4]];
        m[(0, 1)] = 10;
        m[(1, 0)] = 20;
        assert_eq!(m[(0, 1)], 10);
        assert_eq!(m[(1, 0)], 20);
    }

    #[test]
    fn test_mat_macro_from_value() {
        let m = mat![0; 3, 4];
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(2, 3)], 0);

        let m2 = mat![1.0; 2, 2];
        assert_eq!(m2.rows(), 2);
        assert_eq!(m2.cols(), 2);
        assert_eq!(m2[(0, 0)], 1.0);
    }

    #[test]
    fn test_mat_macro_nested() {
        let m = mat![[1, 2, 3], [4, 5, 6]];
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 2)], 6);

        let m2 = mat![[1, 2, 3], [4, 5, 6],];
        assert_eq!(m2.rows(), 2);
        assert_eq!(m2.cols(), 3);
    }

    #[test]
    fn test_add() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];

        let result = &m1 + &m2;
        assert_eq!(result, mat![[6, 8], [10, 12]]);

        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];

        let result = &m1 + m2;
        assert_eq!(result, mat![[6, 8], [10, 12]]);

        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];

        let result = m1 + &m2;
        assert_eq!(result, mat![[6, 8], [10, 12]]);

        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];

        let result = m1 + m2;
        assert_eq!(result, mat![[6, 8], [10, 12]]);
    }

    #[test]
    #[should_panic]
    fn test_add_dimension_mismatch() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[1, 2, 3]];
        let _result = &m1 + &m2;
    }

    #[test]
    fn test_sub() {
        let m1 = mat![[5, 6], [7, 8]];
        let m2 = mat![[1, 2], [3, 4]];
        let result = &m1 - &m2;
        assert_eq!(result, mat![[4, 4], [4, 4]]);

        let m1 = mat![[5, 6], [7, 8]];
        let m2 = mat![[1, 2], [3, 4]];
        let result = &m1 - m2;
        assert_eq!(result, mat![[4, 4], [4, 4]]);
        let m1 = mat![[5, 6], [7, 8]];
        let m2 = mat![[1, 2], [3, 4]];
        let result = m1 - &m2;
        assert_eq!(result, mat![[4, 4], [4, 4]]);
        let m1 = mat![[5, 6], [7, 8]];
        let m2 = mat![[1, 2], [3, 4]];
        let result = m1 - m2;
        assert_eq!(result, mat![[4, 4], [4, 4]]);
    }

    #[test]
    fn test_dot() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];
        let result = m1.dot(&m2);
        assert_eq!(result, mat![[19, 22], [43, 50]]);
    }

    #[test]
    fn test_dot_non_square() {
        let m1 = mat![[1, 2, 3], [4, 5, 6]];
        let m2 = mat![[7, 8], [9, 10], [11, 12]];
        let result = m1.dot(&m2);
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        assert_eq!(result, mat![[58, 64], [139, 154]]);
    }

    #[test]
    #[should_panic]
    fn test_dot_dimension_mismatch() {
        let m1 = mat![[1, 2]];
        let m2 = mat![[3, 4]];
        let _result = m1.dot(&m2);
    }

    #[test]
    fn test_mul_operator() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];
        let result = &m1 * &m2;
        assert_eq!(result, mat![[19, 22], [43, 50]]);
    }

    #[test]
    fn test_component_mul() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[5, 6], [7, 8]];
        let result = m1.component_mul(&m2);
        assert_eq!(result, mat![[5, 12], [21, 32]]);
    }

    #[test]
    #[should_panic]
    fn test_component_mul_dimension_mismatch() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[1, 2, 3]];
        let _result = m1.component_mul(&m2);
    }

    #[test]
    fn test_transpose() {
        let m = mat![[1, 2, 3], [4, 5, 6]];
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t, mat![[1, 4], [2, 5], [3, 6]]);
    }

    #[test]
    fn test_transpose_square() {
        let m = mat![[1, 2], [3, 4]];
        let t = m.transpose();
        assert_eq!(t, mat![[1, 3], [2, 4]]);
    }

    #[test]
    fn test_transpose_twice() {
        let m = mat![[1, 2, 3], [4, 5, 6]];
        let t = m.transpose().transpose();
        assert_eq!(t, m);
    }

    #[test]
    fn test_norm() {
        let m = mat![[3.0, 4.0]];
        assert_eq!(m.norm(), 5.0);

        let m2 = mat![[1.0, 2.0], [2.0, 3.0]];
        let expected = (1.0 + 4.0 + 4.0 + 9.0_f64).sqrt();
        assert!((m2.norm() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_clone() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = m1.clone();
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_partial_eq() {
        let m1 = mat![[1, 2], [3, 4]];
        let m2 = mat![[1, 2], [3, 4]];
        let m3 = mat![[1, 2], [3, 5]];
        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_display() {
        let m = mat![[1, 2], [3, 4]];
        let display_str = format!("{}", m);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_debug() {
        let m = mat![[1, 2], [3, 4]];
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("Matrix"));
    }
}
