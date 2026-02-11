use std::ops::Neg;

use num_complex::Complex;

use super::{Mat, MatMut, MatRef};

impl<'a, T: Clone> MatRef<'a, Complex<T>> {
    pub fn re(self) -> Mat<T> {
        self.map(|x| x.re.clone())
    }

    pub fn im(self) -> Mat<T> {
        self.map(|x| x.im.clone())
    }
}

impl<'a, T: Clone + num_traits::Num + Neg<Output = T>> MatRef<'a, Complex<T>> {
    pub fn conj(self) -> Mat<Complex<T>> {
        self.map(|x| x.conj())
    }

    pub fn adjoint(self) -> Mat<Complex<T>> {
        self.transpose().conj()
    }

    pub fn is_hermitian(self) -> bool
    where
        T: PartialEq,
    {
        if !self.is_square() {
            return false;
        }
        for j in 0..self.ncols() {
            for i in 0..=j {
                if *self.at(i, j) != self.at(j, i).conj() {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T: num_traits::Float> MatRef<'a, Complex<T>> {
    pub fn norm(self) -> Mat<T> {
        self.map(|x| x.norm())
    }

    pub fn norm_sqr(self) -> Mat<T> {
        self.map(|x| x.norm_sqr())
    }

    pub fn arg(self) -> Mat<T> {
        self.map(|x| x.arg())
    }

    pub fn powc(self, n: Complex<T>) -> Mat<Complex<T>> {
        self.map(|x| x.powc(n))
    }

    pub fn powf(self, n: T) -> Mat<Complex<T>> {
        self.map(|x| x.powf(n))
    }

    pub fn log(self, base: T) -> Mat<Complex<T>> {
        self.map(|x| x.log(base))
    }
}

impl<'a, T: Clone> MatMut<'a, Complex<T>> {
    pub fn re(&self) -> Mat<T> {
        self.rb().re()
    }

    pub fn im(&self) -> Mat<T> {
        self.rb().im()
    }
}

impl<T: Clone + num_traits::Num + Neg<Output = T>> MatMut<'_, Complex<T>> {
    pub fn conj(&self) -> Mat<Complex<T>> {
        self.rb().conj()
    }

    pub fn adjoint(&self) -> Mat<Complex<T>> {
        self.rb().adjoint()
    }

    pub fn is_hermitian(&self) -> bool
    where
        T: PartialEq,
    {
        self.rb().is_hermitian()
    }
}

impl<T: num_traits::Float> MatMut<'_, Complex<T>> {
    pub fn norm(&self) -> Mat<T> {
        self.rb().norm()
    }

    pub fn norm_sqr(&self) -> Mat<T> {
        self.rb().norm_sqr()
    }

    pub fn arg(&self) -> Mat<T> {
        self.rb().arg()
    }

    pub fn powc(&self, n: Complex<T>) -> Mat<Complex<T>> {
        self.rb().powc(n)
    }

    pub fn powf(&self, n: T) -> Mat<Complex<T>> {
        self.rb().powf(n)
    }

    pub fn log(&self, base: T) -> Mat<Complex<T>> {
        self.rb().log(base)
    }
}

impl<T: Clone> Mat<Complex<T>> {
    pub fn re(&self) -> Mat<T> {
        self.as_ref().re()
    }

    pub fn im(&self) -> Mat<T> {
        self.as_ref().im()
    }
}

impl<T: Clone + num_traits::Num + Neg<Output = T>> Mat<Complex<T>> {
    pub fn conj(&self) -> Mat<Complex<T>> {
        self.as_ref().conj()
    }

    pub fn adjoint(&self) -> Mat<Complex<T>> {
        self.as_ref().adjoint()
    }

    pub fn is_hermitian(&self) -> bool
    where
        T: PartialEq,
    {
        self.as_ref().is_hermitian()
    }
}

impl<T: num_traits::Float> Mat<Complex<T>> {
    pub fn norm(&self) -> Mat<T> {
        self.as_ref().norm()
    }

    pub fn norm_sqr(&self) -> Mat<T> {
        self.as_ref().norm_sqr()
    }

    pub fn arg(&self) -> Mat<T> {
        self.as_ref().arg()
    }

    pub fn powc(&self, n: Complex<T>) -> Mat<Complex<T>> {
        self.as_ref().powc(n)
    }

    pub fn powf(&self, n: T) -> Mat<Complex<T>> {
        self.as_ref().powf(n)
    }

    pub fn log(&self, base: T) -> Mat<Complex<T>> {
        self.as_ref().log(base)
    }
}
