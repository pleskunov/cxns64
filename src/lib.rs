//! # `cxns64`
//!
//! ComplexNumbers is a simple library that implements support for
//! 64-bit floating-point complex numbers. It implements all basic 
//! operations such as addition, subtraction, multiplication and
//! division.


/// A 64-bit floating-point complex number in Cartesian form
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 { 
    /// Create a new complex number
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    
    /// A complex conjugate of the complex number
    #[inline]
    pub fn conj(self) -> Self {
       Self { re: self.re, im: -self.im } 
    }

    /// A square of Euclidian norm of the complex number
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Get a real part of the complex number 
    #[inline]
    pub fn real(self) -> f64 {
        self.re
    }

    /// Get an imaginary part of the complex number
    #[inline]
    pub fn imag(self) -> f64 {
        self.im
    }

    /// Multiply `Self` with a scalar
    #[inline]
    pub fn upscale(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }

    /// Divide `self` by a scalar
    #[inline]
    pub fn downscale(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

use core::ops::{Add, Sub, Mul, Div};

// (a + i b) + (c + i d) == (a + c) + i (b + d)
impl Add for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.re + rhs.re, self.im + rhs.im)
    }
}

// (a + i b) - (c + i d) == (a - c) + i (b - d)
impl Sub for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.re - rhs.re, self.im - rhs.im)
    }
}

// (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
impl Mul for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let re: f64 = self.re * rhs.re - self.im * rhs.im;
        let im: f64 = self.re * rhs.im + self.im * rhs.re;
        Self::Output::new(re, im)
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl Div for Complex64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let norm_sqr = rhs.norm_sqr();
        let re = (self.re * rhs.re + self.im * rhs.im) / norm_sqr;
        let im = (self.im * rhs.re - self.re * rhs.im) / norm_sqr;
        Self::Output::new(re, im)
    }
}


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
