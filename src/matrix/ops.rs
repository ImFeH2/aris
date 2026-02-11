use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_complex::Complex;

use super::{Mat, MatMut, MatRef};

macro_rules! impl_mat_mat_binop {
    ($OpTrait:ident, $op_fn:ident) => {
        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatRef<'_, T>> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatRef<'_, T>) -> Mat<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "shape mismatch: {:?} vs {:?}",
                    self.shape(),
                    rhs.shape()
                );
                let (nrows, ncols) = self.shape();
                let mut data = Vec::with_capacity(self.size());
                for j in 0..ncols {
                    for i in 0..nrows {
                        data.push($OpTrait::$op_fn(
                            self.at(i, j).clone(),
                            rhs.at(i, j).clone(),
                        ));
                    }
                }
                Mat::from_vec_col(nrows, ncols, data)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&Mat<T>> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait for Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&Mat<T>> for Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<Mat<T>> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&Mat<T>> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self, rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatRef<'_, T>> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatRef<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatRef<'_, T>> for Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatRef<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<Mat<T>> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self, rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatRef<'_, T>> for MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatRef<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatMut<'_, T>> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self, rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatMut<'_, T>> for MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatRef<'_, T>> for &MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatRef<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&MatMut<'_, T>> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self, rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&MatMut<'_, T>> for &MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&Mat<T>> for MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatMut<'_, T>> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&Mat<T>> for &MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<&MatMut<'_, T>> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: &MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.rb())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<Mat<T>> for MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: Mat<T>) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs.as_ref())
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<MatMut<'_, T>> for Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: MatMut<'_, T>) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs.rb())
            }
        }
    };
}

impl_mat_mat_binop!(Add, add);
impl_mat_mat_binop!(Sub, sub);

macro_rules! impl_scalar_rmul {
    ($OpTrait:ident, $op_fn:ident) => {
        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<T> for MatRef<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: T) -> Mat<T> {
                let (nrows, ncols) = self.shape();
                let mut data = Vec::with_capacity(self.size());
                for j in 0..ncols {
                    for i in 0..nrows {
                        data.push($OpTrait::$op_fn(self.at(i, j).clone(), rhs.clone()));
                    }
                }
                Mat::from_vec_col(nrows, ncols, data)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<T> for &Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: T) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<T> for Mat<T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: T) -> Mat<T> {
                $OpTrait::$op_fn(self.as_ref(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<T> for MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: T) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs)
            }
        }

        impl<T: Clone + $OpTrait<Output = T>> $OpTrait<T> for &MatMut<'_, T> {
            type Output = Mat<T>;
            fn $op_fn(self, rhs: T) -> Mat<T> {
                $OpTrait::$op_fn(self.rb(), rhs)
            }
        }
    };
}

impl_scalar_rmul!(Mul, mul);
impl_scalar_rmul!(Div, div);

macro_rules! impl_scalar_lmul {
    ($scalar:ty) => {
        impl Mul<MatRef<'_, $scalar>> for $scalar {
            type Output = Mat<$scalar>;
            fn mul(self, rhs: MatRef<'_, $scalar>) -> Mat<$scalar> {
                Mul::mul(rhs, self)
            }
        }

        impl Mul<&Mat<$scalar>> for $scalar {
            type Output = Mat<$scalar>;
            fn mul(self, rhs: &Mat<$scalar>) -> Mat<$scalar> {
                Mul::mul(rhs.as_ref(), self)
            }
        }

        impl Mul<Mat<$scalar>> for $scalar {
            type Output = Mat<$scalar>;
            fn mul(self, rhs: Mat<$scalar>) -> Mat<$scalar> {
                Mul::mul(rhs.as_ref(), self)
            }
        }

        impl Mul<MatMut<'_, $scalar>> for $scalar {
            type Output = Mat<$scalar>;
            fn mul(self, rhs: MatMut<'_, $scalar>) -> Mat<$scalar> {
                Mul::mul(rhs.rb(), self)
            }
        }

        impl Mul<&MatMut<'_, $scalar>> for $scalar {
            type Output = Mat<$scalar>;
            fn mul(self, rhs: &MatMut<'_, $scalar>) -> Mat<$scalar> {
                Mul::mul(rhs.rb(), self)
            }
        }
    };
}

impl_scalar_lmul!(f32);
impl_scalar_lmul!(f64);
impl_scalar_lmul!(i8);
impl_scalar_lmul!(i16);
impl_scalar_lmul!(i32);
impl_scalar_lmul!(i64);
impl_scalar_lmul!(i128);
impl_scalar_lmul!(u8);
impl_scalar_lmul!(u16);
impl_scalar_lmul!(u32);
impl_scalar_lmul!(u64);
impl_scalar_lmul!(u128);
impl_scalar_lmul!(isize);
impl_scalar_lmul!(usize);
impl_scalar_lmul!(Complex<f32>);
impl_scalar_lmul!(Complex<f64>);

impl<T: Clone + Neg<Output = T>> Neg for MatRef<'_, T> {
    type Output = Mat<T>;
    fn neg(self) -> Mat<T> {
        let (nrows, ncols) = self.shape();
        let mut data = Vec::with_capacity(self.size());
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(Neg::neg(self.at(i, j).clone()));
            }
        }
        Mat::from_vec_col(nrows, ncols, data)
    }
}

impl<T: Clone + Neg<Output = T>> Neg for &Mat<T> {
    type Output = Mat<T>;
    fn neg(self) -> Mat<T> {
        Neg::neg(self.as_ref())
    }
}

impl<T: Clone + Neg<Output = T>> Neg for Mat<T> {
    type Output = Mat<T>;
    fn neg(self) -> Mat<T> {
        Neg::neg(self.as_ref())
    }
}

impl<T: Clone + Neg<Output = T>> Neg for MatMut<'_, T> {
    type Output = Mat<T>;
    fn neg(self) -> Mat<T> {
        Neg::neg(self.rb())
    }
}

impl<T: Clone + Neg<Output = T>> Neg for &MatMut<'_, T> {
    type Output = Mat<T>;
    fn neg(self) -> Mat<T> {
        Neg::neg(self.rb())
    }
}

macro_rules! impl_mat_assign_op {
    ($OpTrait:ident, $op_fn:ident) => {
        impl<T: Clone + $OpTrait> $OpTrait<MatRef<'_, T>> for MatMut<'_, T> {
            fn $op_fn(&mut self, rhs: MatRef<'_, T>) {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "shape mismatch: {:?} vs {:?}",
                    self.shape(),
                    rhs.shape()
                );
                let (nrows, ncols) = self.shape();
                for j in 0..ncols {
                    for i in 0..nrows {
                        $OpTrait::$op_fn(self.at_mut(i, j), rhs.at(i, j).clone());
                    }
                }
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<&Mat<T>> for MatMut<'_, T> {
            fn $op_fn(&mut self, rhs: &Mat<T>) {
                $OpTrait::$op_fn(self, rhs.as_ref());
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<&MatMut<'_, T>> for MatMut<'_, T> {
            fn $op_fn(&mut self, rhs: &MatMut<'_, T>) {
                $OpTrait::$op_fn(self, rhs.rb());
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<MatRef<'_, T>> for Mat<T> {
            fn $op_fn(&mut self, rhs: MatRef<'_, T>) {
                $OpTrait::$op_fn(&mut self.as_mut(), rhs);
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<&Mat<T>> for Mat<T> {
            fn $op_fn(&mut self, rhs: &Mat<T>) {
                $OpTrait::$op_fn(&mut self.as_mut(), rhs.as_ref());
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<Mat<T>> for Mat<T> {
            fn $op_fn(&mut self, rhs: Mat<T>) {
                $OpTrait::$op_fn(&mut self.as_mut(), rhs.as_ref());
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<&MatMut<'_, T>> for Mat<T> {
            fn $op_fn(&mut self, rhs: &MatMut<'_, T>) {
                $OpTrait::$op_fn(&mut self.as_mut(), rhs.rb());
            }
        }
    };
}

impl_mat_assign_op!(AddAssign, add_assign);
impl_mat_assign_op!(SubAssign, sub_assign);

macro_rules! impl_scalar_assign_op {
    ($OpTrait:ident, $op_fn:ident) => {
        impl<T: Clone + $OpTrait> $OpTrait<T> for MatMut<'_, T> {
            fn $op_fn(&mut self, rhs: T) {
                let (nrows, ncols) = self.shape();
                for j in 0..ncols {
                    for i in 0..nrows {
                        $OpTrait::$op_fn(self.at_mut(i, j), rhs.clone());
                    }
                }
            }
        }

        impl<T: Clone + $OpTrait> $OpTrait<T> for Mat<T> {
            fn $op_fn(&mut self, rhs: T) {
                $OpTrait::$op_fn(&mut self.as_mut(), rhs);
            }
        }
    };
}

impl_scalar_assign_op!(MulAssign, mul_assign);
impl_scalar_assign_op!(DivAssign, div_assign);
