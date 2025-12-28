//! # `cxns64`
//!
//! A lightweight Rust library providing a fixed-precision (`f64`) complex number
//! type with basic arithmetic operations.
//!
//! The crate implements addition, subtraction, multiplication, division,
//! and selected elementary functions for complex numbers.
//!
//! ## Compatibility
//!
//! The `cxns64` crate is tested with rustc 1.92 and newer.

const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-9;

#[inline]
fn is_zero(x: f64) -> bool {
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

    /// Create an imaginary unit
    #[inline]
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
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

    /// e^(self)
    /// e ^ (a + i b) = e^a * (cos(b) + i sin(b)) 
    #[inline]
    pub fn exp(self) -> Self {
        let ex = self.re.exp();
        Self { re: ex * self.im.cos(), im: ex * self.im.sin() }
    }

    /// TODO: check
    #[inline]
    pub fn log(self) -> Self {
        Self { re: self.re.hypot(self.im).ln(), im: self.im.atan2(self.re) }
    }


    /// A real part of `self` 
    #[inline]
    pub fn real(self) -> f64 {
        self.re
    }

    /// An imaginary part of `self`
    #[inline]
    pub fn imag(self) -> f64 {
        self.im
    }

    /// Check if the complex number is NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    
    /// Check if the complex number is infinite
    #[inline]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Check if the complex numer is finite
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

/// Comparison operators: == and !=
impl PartialEq for Complex64 {
    fn eq(&self, rhs: &Self) -> bool {
        is_equal(self.re, rhs.re) && is_equal(self.im, rhs.im)
    }
}

/// Addition, subtraction, multiplication and division operators
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

// c64 * f64
impl Mul<f64> for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

// f64 * f64 + f64 fused
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
    
    /// Fused multiply add: `self` + `rhs` + `acc`
    #[inline]
    fn fma(self, rhs: Self, acc: Self) -> Self::Output {
        let re: f64 = self.im.mul_add(-rhs.im, self.re.mul_add(rhs.re, acc.re));
        let im: f64 = self.re.mul_add(rhs.im, self.im.mul_add(rhs.re, acc.im));
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

// c64 / f64
impl Div<f64> for Complex64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

/// Addition, subtraction, multiplication and division assignment operators
use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

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

/// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor() {
        let z = Complex64::new(2.0, 5.0);
        assert_eq!(z.re, 2.0);
        assert_eq!(z.im, 5.0);
    }

    #[test]
    fn partial_eq() {
        let a = Complex64::new(-4.3, 6.7);
        let b = Complex64::new(-4.3, 6.7);
        assert!(a == b)
    }

    #[test]
    fn add() {
        let a = Complex64::new(3.0, 4.0);
        let b = Complex64::new(7.0, 3.0);
        assert_eq!(a + b, Complex64::new(10.0, 7.0))
    }

    #[test]
    fn sub() {
        let a = Complex64::new(6.0, 4.0);
        let b = Complex64::new(2.0, 1.0);
        assert_eq!(a - b, Complex64::new(4.0, 3.0))
    }

    #[test]
    fn mult() {
        let a = Complex64::new(3.0, 8.5);
        let b = Complex64::new(4.1, 2.2);
        assert_eq!(a * b, Complex64::new(-6.4, 41.45))
    }

    #[test]
    fn div() {
        let a = Complex64::new(38.2, 49.5);
        let b = Complex64::new(12.4, 10.0);
        assert!(a / b == Complex64::new(3.81730769, 0.913461538))
    }

    #[test]
    fn add_assign() {
        let mut a = Complex64::new(1.25, -3.5);
        let b = Complex64::new(-2.75, 4.125);
        a += b;
        assert!(a == Complex64::new(-1.5, 0.625));
    }

    
    #[test]
    fn sub_assign() {
        let mut a = Complex64::new(5.75, 2.0);
        let b = Complex64::new(1.25, -3.5);
        a -= b;
        assert!(a == Complex64::new(4.5, 5.5));
    }


    #[test]
    fn mul_assign() {
        let mut a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(4.0, -1.5);
        a *= b;
        assert!(a == Complex64::new(12.5, 9.0));
    }


    #[test]
    fn div_assign() {
        let mut a = Complex64::new(7.5, 2.5);
        let b = Complex64::new(1.25, -0.75);
        a /= b;
        println!("{}, {}", a.re, a.im);
        assert!(a == Complex64::new(3.52941176470, 4.1176470588));
    }

    #[test]
    fn fma() {
        let z = Complex64::new(1.75, -2.25);
        let w = Complex64::new(-0.5, 3.125);
        let a = Complex64::new(4.0, -1.5);

        let fma_result = z.fma(w, a);
       
        let tmp = z * w;
        let reference = tmp + a;

        assert!(fma_result == reference);
    }

    #[test]
    fn fma_rounding_sensitive() {
        let z = Complex64::new(1.0e16, 1.0);
        let w = Complex64::new(1.0e-16, -1.0);
        let a = Complex64::new(1.0, -1.0);

        let fma_result = z.fma(w, a);
        
        let tmp = z * w; 
        let reference = tmp + a;

        assert!(fma_result == reference);
    }

    #[test]
    fn conj() {
       let a = Complex64::new(2.0, 5.0);
       assert!(a.conj() == Complex64::new(2.0, -5.0))
    }

    #[test]
    fn i_unit() {
        let a = Complex64::i();
        assert!(a == Complex64::new(0.0, 1.0))
    }

    #[test]
    fn real() {
        let a = Complex64::new(1.0, 2.0);
        assert_eq!(a.real(), 1.0);
    }

    #[test]
    fn imag() {
        let a = Complex64::new(1.0, 2.0);
        assert_eq!(a.imag(), 2.0);
    }

    #[test]
    fn is_nan() {
        let a = Complex64::new(f64::NAN, 5.0);
        let b = Complex64::new(1.0, f64::NAN);
        assert!(a.is_nan() && b.is_nan())
    }

    #[test]
    fn is_infinite() {
        let a = Complex64::new(f64::INFINITY, 5.0);
        let b = Complex64::new(1.0, f64::INFINITY);
        assert!(a.is_infinite() && b.is_infinite())
    }

    #[test]
    fn is_finite() {
        let a = Complex64::new(f64::INFINITY, 5.0);
        let b = Complex64::new(1.0, 2.0);
        assert!(!a.is_finite() && b.is_finite())
    }
}
