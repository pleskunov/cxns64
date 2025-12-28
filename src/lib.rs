//! # `cxns64`
//!
//! ComplexNumbers is a simple library that implements support for
//! 64-bit floating-point complex numbers. It implements all basic 
//! operations such as addition, subtraction, multiplication and
//! division.

const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-9;

#[inline]
pub fn is_zero(x: f64) -> bool {
    x.abs() <= ATOL
}

#[inline]
fn is_equal(x: f64, y: f64) -> bool {
    let diff: f64 = (x - y).abs();
    let scale: f64 = x.abs().max(y.abs());
    diff <= ATOL.max(RTOL * scale)
}

#[inline]
fn is_less(x: f64, y: f64) -> bool {
    x < y && !is_equal(x, y)
}


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
    
    /// A complex conjugate of self
    #[inline]
    pub fn conj(self) -> Self {
       Self { re: self.re, im: -self.im } 
    }

    /// An Euclidian norm of self (the same as abs()) 
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// An absolute value of self (the same as norm())
    pub fn abs(self) -> f64 {
        self.norm()
    }

    /// A square of Euclidian norm of self
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// A principal Arg of self
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// A real part of self 
    #[inline]
    pub fn real(self) -> f64 {
        self.re
    }

    /// An imaginary part of self
    #[inline]
    pub fn imag(self) -> f64 {
        self.im
    }
}


use core::ops::{Add, Sub, Mul, Div};

pub trait MulAdd<Rhs = Self, Acc = Self> {
    type Output;
    fn fma(self, rhs: Rhs, acc: Acc) -> Self::Output;
}

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

// Allows for c64 * f64
impl Mul<f64> for Complex64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl MulAdd for f64 {
    type Output = f64;

    #[inline(always)]
    fn fma(self, rhs: f64, acc: f64) -> f64 {
        self.mul_add(rhs, acc)
    }
}

// (a + i b) * (c + i d) + (e + i f) == ((a*c + e) - b*d) + i (a*d + (b*c + f))
impl MulAdd for Complex64 {
    type Output = Self;

    #[inline]
    fn fma(self, rhs: Self, acc: Self) -> Self::Output {
        // TODO: partially fused semantics
        let re: f64 = self.re.mul_add(rhs.re, acc.re) - (self.im * rhs.im);
        let im: f64 = self.im.mul_add(rhs.re, acc.im) + (self.re * rhs.im);
        Self::new(re, im)
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl Div for Complex64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        
        debug_assert!(!is_zero(rhs.re) || !is_zero(rhs.im));

        let re: f64;
        let im: f64;
        let r: f64;
        let denom: f64;

        if is_less(rhs.im.abs(), rhs.re.abs()) || is_equal(rhs.re.abs(), rhs.im.abs()) {
            r = rhs.im / rhs.re;
            denom = rhs.re + r * rhs.im;
            re = (self.re + r * self.im) / denom;
            im = (self.im - r * self.re) / denom;
        } else {
            r = rhs.re / rhs.im;
            denom = rhs.im + r * rhs.re;
            re = (self.re * r + self.im) / denom;
            im = (self.im * r - self.re) / denom;
        }

        Self::Output::new(re, im)
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}


use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

impl AddAssign for Complex64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl SubAssign for Complex64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

// (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
impl MulAssign for Complex64 {
    fn mul_assign(&mut self, rhs: Self) {
        let a: f64 = self.re;

        self.re *= rhs.re;
        self.re -= self.im * rhs.im;

        self.im *= rhs.re;
        self.im += a * rhs.im;
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//  == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl DivAssign for Complex64 {
    fn div_assign(&mut self, rhs: Self) {
        
        debug_assert!(!is_zero(rhs.re) || !is_zero(rhs.im));

        let re: f64;
        let im: f64;
        let r: f64;
        let denom: f64;

        if is_less(rhs.im.abs(), rhs.re.abs()) || is_equal(rhs.re.abs(), rhs.im.abs()) {
            r = rhs.im / rhs.re;
            denom = rhs.re + r * rhs.im;
            re = (self.re + r * self.im) / denom;
            im = (self.im - r * self.re) / denom;
        } else {
            r = rhs.re / rhs.im;
            denom = rhs.im + r * rhs.re;
            re = (self.re * r + self.im) / denom;
            im = (self.im * r - self.re) / denom;
        }

        self.re = re;
        self.im = im;
    }
}

impl PartialEq for Complex64 {
    fn eq(&self, other: &Self) -> bool {
        is_equal(self.re, other.re) && is_equal(self.im, other.im)
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
