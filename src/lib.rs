//! # `cxns64`
//!
//! A lightweight Rust library providing a fixed-precision (`f64`) complex number
//! type with basic arithmetic operations.

/// MIT License

/// Copyright (c) 2026 Pavel Pleskunov

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// Approximately two machine epsilons around 1.0.
#[allow(unused)]
pub const ULP2_EPS: f64 = 2.0_f64 * f64::EPSILON;

/// Approximately 128 machine epsilons around 1.0.
#[allow(unused)]
pub const ULP128_EPS: f64 = 128.0_f64 * f64::EPSILON; 

/// A 64-bit floating-point complex number in Cartesian form
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 { 
    /// Constructor
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    // Constructor for a complex number with its real part set to 1.0
    #[inline(always)]
    pub fn re_unit() -> Self {
        Self::new(1.0, 0.0)
    }

    // Constructor for a complex number with its imaginary part set to 1.0
    #[inline(always)]
    pub fn im_unit() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Constructor for an imaginary unit
    // This is alias for `Complex64::im_unit()`.
    #[inline(always)]
    pub fn i() -> Self {
        Self::im_unit()
    }

    /// Multiply `self` by the scalar `rhs`
    #[inline(always)]
    pub fn scale(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }

    /// A complex conjugate of `self`
    #[inline(always)]
    pub fn conj(self) -> Self {
       Self { re: self.re, im: -self.im } 
    }

    /// An absolute value of `self`
    #[inline(always)]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)   
    }

    /// Squared magnitude of `self`.
    /// The norm calculated by this function is also 
    /// known as field norm or absolute square.
    #[inline(always)]
    pub fn norm(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Check if `self` is exactly zero
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.re == 0.0_f64 && self.im == 0.0_f64
    }

    /// A principal square root of `self`
    #[inline]
    pub fn sqrt(self) -> Self { 
        let re0: f64 = self.re;
        let im0: f64 = self.im;

        if re0 == 0.0_f64 && im0 == 0.0_f64 {
            return Self::new(0.0, 0.0)
        }

        let x: f64 = re0.abs();
        let y: f64 = im0.abs();

        let w: f64 = match x >= y {
            true => {  
                let r: f64 = y / x;
                x.sqrt() * (0.5 * (1.0 + (1.0 + r * r).sqrt())).sqrt()
            },
            false => {
                let r: f64 = x / y;
                y.sqrt() * (0.5 * (r + (1.0 + r * r).sqrt())).sqrt()
            },
        };

        let (re, im) = match re0 >= 0.0 {
            true => (w, im0 / (2.0 * w)),
            false => { 
                let im = match im0 >= 0.0 { true => w, false => -w, };
                (im0 / (2.0 * im), im) 
            }
        };

        Self::new(re, im)
    }

    /// The sine of `self`
    #[inline(always)]
    pub fn sin(self) -> Self {
        // sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        Self::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }

    /// The cosine of `self`
    #[inline(always)]
    pub fn cos(self) -> Self {
        // cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        Self::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }

    /// A real part of `self` 
    #[inline(always)]
    pub fn real(self) -> f64 {
        self.re
    }

    /// An imaginary part of `self`
    #[inline(always)]
    pub fn imag(self) -> f64 {
        self.im
    }

    /// Check if `self` is NaN
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Check if `self` is infinite
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Check if `self` is finite
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
 
    /// Comparison using the relative and absolute epsilon.
    ///
    /// # Arguments
    /// * `rhs` - The other complex number instance.
    /// * `rel_eps` - The relative error margin.
    /// * `abs_eps` - The absolute error margin.
    ///
    /// # Returns
    /// True if `self` is approximately equal `rhs` and false otherwise.
    ///
    /// # Notes
    /// Two complex numbers are approximately equal if their real and 
    /// imaginary components are approximately equal independently.
    #[inline]
    pub fn approx_eq(self, rhs: Self, abs_eps: f64, rel_eps: f64) -> bool {    
        debug_assert!(rel_eps >= f64::EPSILON);
        debug_assert!(rel_eps < 1.0_f64);
        debug_assert!(abs_eps > 0.0_f64);

        if self.is_nan() || rhs.is_nan() {
            return false;
        }

        if self.re == rhs.re && self.im == rhs.im {
            return true;
        }

        if !self.is_finite() || !rhs.is_finite() {
            return false;
        }

        let re_diff = (self.re - rhs.re).abs();
        let im_diff = (self.im - rhs.im).abs();

        let re_norm = self.re.abs().max(rhs.re.abs());
        let im_norm = self.im.abs().max(rhs.im.abs());

        let re_scaled_eps = rel_eps * re_norm;
        let im_scaled_eps = rel_eps * im_norm;

        re_diff <= abs_eps.max(re_scaled_eps) && im_diff <= abs_eps.max(im_scaled_eps)
    }

    /// Comparison against zero using an absolute error margin (epsilon).
    ///
    /// # Arguments
    /// * `abs_eps` - The absolute error margin.
    ///
    /// # Returns
    /// True if `self` is approximately (less or equal to the error margin) equal 0.0 and false otherwise.
    ///
    /// # Notes
    /// - Typically, setting `abs_eps` to `f64::EPSILON`, or some small multiple 
    /// of `f64::EPSILON` works well. Make it larger if greater error is expected.
    #[inline]
    pub fn approx_zero(self, abs_eps: f64) -> bool {
        debug_assert!(abs_eps > 0.0_f64);

        if self.is_nan() {
            return false;
        }

        if !self.is_finite() {
            return false;
        }

        self.re.abs() <= abs_eps && self.im.abs() <= abs_eps
    }

    /// This a convenience wrapper over `approx_eq()` comparator
    /// that uses `ULP128_EPS` as absolute epsilon and `ULP2_EPS`
    /// as the relative epsilon.
    #[inline(always)]
    pub fn approx_eq_def(self, rhs: Self) -> bool {
        self.approx_eq(rhs, ULP128_EPS, ULP2_EPS)
    }

    /// This a convenience wrapper over `approx_zero()` comparator
    /// that uses `ULP128_EPS` as absolute epsilon.
    #[inline(always)]
    pub fn approx_zero_def(self) -> bool {
        self.approx_zero(ULP128_EPS)
    }
}

/// Comparison operators: == and !=
impl PartialEq for Complex64 {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        self.re == rhs.re && self.im == rhs.im
    }
}

/// Addition, subtraction, multiplication and division operators
use core::ops::{Add, Sub, Mul, Div, Neg};

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

// f64 * c64
impl Mul<Complex64> for f64 {
    type Output = Complex64;

    #[inline(always)]
    fn mul(self, rhs: Complex64) -> Self::Output {
        Self::Output::new(rhs.re * self, rhs.im * self)
    }
}

// c64 * f64
impl Mul<f64> for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output::new(self.re * rhs, self.im * rhs)
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
        debug_assert!(!rhs.is_zero());

        let re: f64;
        let im: f64;
        let r: f64;
        let denom: f64;

        if rhs.im.abs() <= rhs.re.abs() {
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

// c64 = -c64
impl Neg for Complex64 {
    type Output = Self;
    
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::Output::new(-self.re, -self.im)
    }
}

// c64 / f64
impl Div<f64> for Complex64 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

/// Addition, subtraction, multiplication and division assignment operators
use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

impl AddAssign for Complex64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl SubAssign for Complex64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

// (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
impl MulAssign for Complex64 {
    #[inline]
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
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        debug_assert!(!rhs.is_zero());

        let re: f64;
        let im: f64;
        let r: f64;
        let denom: f64;
        
        if rhs.im.abs() <= rhs.re.abs() {
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

/// Unit tests

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn is_approx_equal(x: f64, y: f64, abs_eps: f64, rel_eps: f64) -> bool {
        let diff = (x - y).abs();
        let norm = x.abs().max(y.abs());
        diff <= abs_eps.max(rel_eps * norm)
    }

    #[inline]
    fn is_real_dominant(b: Complex64) -> bool {
        b.re.abs() >= b.im.abs()
    }

    #[allow(non_upper_case_globals)]
    const _0_0i: Complex64 = Complex64 { re: 0.0, im: 0.0 };
    #[allow(non_upper_case_globals)]
    const _1_0i: Complex64 = Complex64 { re: 1.0, im: 0.0 };
    #[allow(non_upper_case_globals)]
    const _1_1i: Complex64 = Complex64 { re: 1.0, im: 1.0 };
    #[allow(non_upper_case_globals)]
    const _0_1i: Complex64 = Complex64 { re: 0.0, im: 1.0 };
    #[allow(non_upper_case_globals)]
    const _neg1_1i: Complex64 = Complex64 { re: -1.0, im: 1.0 };
    #[allow(non_upper_case_globals)]
    const _05_05i: Complex64 = Complex64 { re: 0.5, im: 0.5 };

    const ALL_CONSTS: [Complex64; 5] = [
        _0_0i, _1_0i, _1_1i, _neg1_1i, _05_05i,
    ];

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
    fn partial_eq_nan() {
        let a = Complex64::new(f64::NAN, 1.0);
        assert!(a != a);
    }

    #[test]
    fn partial_eq_signed_zero() {
        let a = Complex64::new(0.0, 0.0);
        let b = Complex64::new(-0.0, -0.0);
        assert!(a == b);
    }

    #[test]
    fn approx_eq_nan() {
        let a = Complex64::new(f64::NAN, f64::NAN);
        let b = Complex64::new(f64::NAN, 2.0);
        let c = Complex64::new(2.0, f64::NAN);
        assert!(!a.approx_eq(b, ULP128_EPS, ULP2_EPS));
        assert!(!a.approx_eq(a, ULP128_EPS, ULP2_EPS));
        assert!(!a.approx_eq(c, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn approx_eq_infinity() {
        let a = Complex64::new(f64::INFINITY, f64::INFINITY);
        let b = Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let c = Complex64::new(2.0, f64::INFINITY);
        assert!(a.approx_eq(a, ULP128_EPS, ULP2_EPS));
        assert!(b.approx_eq(b, ULP128_EPS, ULP2_EPS));
        assert!(!a.approx_eq(b, ULP128_EPS, ULP2_EPS));
        assert!(!a.approx_eq(c, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn approx_eq_symmetric() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(1.0 + 1e-15, 2.0 - 1e-15);
        assert_eq!(
            a.approx_eq(b, ULP128_EPS, ULP2_EPS),
            b.approx_eq(a, ULP128_EPS, ULP2_EPS)
        );
    }

    #[test]
    fn approx_eq_componentwise() {
        let a = Complex64::new(0.0, 0.0);
        let b = Complex64::new(1e-15, 1e-6);
        assert!(!a.approx_eq(b, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn approx_zero_nan() {
        let a = Complex64::new(f64::NAN, f64::NAN);
        assert!(!a.approx_zero(ULP2_EPS));
    }

    #[test]
    fn approx_zero_infinity() {
        let a = Complex64::new(f64::INFINITY, f64::INFINITY);
        let b = Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!(!a.approx_zero(ULP2_EPS));
        assert!(!b.approx_zero(ULP2_EPS));
    }

    #[test]
    fn approx_zero_exact() {
        let a = Complex64::new(0.0, 0.0);
        assert!(a.approx_zero(ULP2_EPS));
    }

    #[test]
    fn approx_zero_value_lt_epsilon() {
        let a = Complex64::new(1.0e-17_f64, 1.0e-17_f64);
        assert!(a.approx_zero(ULP128_EPS));
    }

    #[test]
    fn approx_zero_value_gt_epsilon() {
        let a = Complex64::new(1.0e-6_f64, 1.0e-6_f64);
        assert!(!a.approx_zero(ULP128_EPS));
    }

    #[test]
    fn approx_eq_equal() {
        let a = Complex64::new(1.0_f64, 1.0_f64);
        let b = Complex64::new(1.0_f64 + 1.0e-17, 1.0_f64 + 1.0e-17);
        assert!(a.approx_eq(b, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn approx_eq_not_equal() {
        let a = Complex64::new(1.0_f64, 1.0_f64);
        let b = Complex64::new(1.0_f64 + 1.0e-6, 1.0_f64 + 1.0e-6);
        assert!(!a.approx_eq(b, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn approx_eq_abs_boundary() {
        let a = Complex64::new(0.0_f64, 0.0_f64);
        let b = Complex64::new(1.0e-14, 1.0e-14);
        assert!(a.approx_eq(b, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn add() {
        let a = Complex64::new(3.0, 4.0);
        let b = Complex64::new(7.0, 3.0);
        let result = a + b;
        let expected = Complex64::new(10.0, 7.0); 
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn sub() {
        let a = Complex64::new(6.0, 4.0);
        let b = Complex64::new(2.0, 1.0);
        let result = a - b;
        let expected = Complex64::new(4.0, 3.0); 
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }
    
    #[test]
    fn add_sub_inverse() {
        let a = Complex64::new(3.5, -2.1);
        let b = Complex64::new(7.0, 4.0);
        assert!(((a + b) - b).approx_eq(a, ULP2_EPS, ULP2_EPS));
    }

    #[test]
    fn mult() {
        let a = Complex64::new(3.0, 8.5);
        let b = Complex64::new(4.1, 2.2);
        let result = a * b;
        let expected = Complex64::new(-6.4, 41.45);
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn div() {
        let a = Complex64::new(38.2, 49.5);
        let b = Complex64::new(12.4, 10.0);
        let expected = Complex64::new(3.8173076923076925, 0.9134615384615383);
        let result = a / b;
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS))
    }

    #[test]
    fn div_inverse_property() {
        let a = Complex64::new(3.2, -1.7);
        let b = Complex64::new(2.1, 0.8);
        let r = a / b;
        assert!(
            (r * b).approx_eq(a, ULP128_EPS, ULP2_EPS)
        );
    }

    #[test]
    fn div_branch_real_dominant() {
        let a = Complex64::new(38.2, 49.5);
        let b = Complex64::new(12.4, 1.0); // |re| >> |im|
        assert!(is_real_dominant(b));
        let result = a / b;
        let expected = Complex64::new(
            (a.re * b.re + a.im * b.im) / b.norm(),
            (a.im * b.re - a.re * b.im) / b.norm(),
        );
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }
    
    #[test]
    fn div_branch_imag_dominant() {
        let a = Complex64::new(38.2, 49.5);
        let b = Complex64::new(1.0, 12.4); // |im| >> |re|
        assert!(!is_real_dominant(b));
        let result = a / b;
        let expected = Complex64::new(
            (a.re * b.re + a.im * b.im) / b.norm(),
            (a.im * b.re - a.re * b.im) / b.norm(),
        );
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }
    
    #[test]
    fn div_branch_boundary_equal_magnitude() {
        let a = Complex64::new(10.0, -3.0);
        let b = Complex64::new(5.0, 5.0); // |re| == |im|
        let result = a / b;
        let expected = Complex64::new(
            (a.re * b.re + a.im * b.im) / b.norm(),
            (a.im * b.re - a.re * b.im) / b.norm(),
        );
        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn div_branch_extreme_scale_separation() {
        let a = Complex64::new(1.0, 2.0);
        let b_small_im = Complex64::new(1e200, 1.0); // branch A
        let b_small_re = Complex64::new(1.0, 1e200); // branch B

        let r1 = a / b_small_im;
        let r2 = a / b_small_re;

        let expected1 = Complex64::new(
            (a.re * b_small_im.re + a.im * b_small_im.im) / b_small_im.norm(),
            (a.im * b_small_im.re - a.re * b_small_im.im) / b_small_im.norm(),
        );

        let expected2 = Complex64::new(
            (a.re * b_small_re.re + a.im * b_small_re.im) / b_small_re.norm(),
            (a.im * b_small_re.re - a.re * b_small_re.im) / b_small_re.norm(),
        );

        assert!(r1.approx_eq(expected1, ULP128_EPS, ULP2_EPS));
        assert!(r2.approx_eq(expected2, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn div_branch_near_real_axis() {
        let a = Complex64::new(1.0, 1.0);
        let b = Complex64::new(1.0, 1e-16); // almost real → branch A

        let result = a / b;

        let expected = Complex64::new(
            (a.re * b.re + a.im * b.im) / b.norm(),
            (a.im * b.re - a.re * b.im) / b.norm(),
        );

        assert!(result.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn add_assign() {
        let mut a = Complex64::new(1.25, -3.5);
        let b = Complex64::new(-2.75, 4.125);
        a += b;
        let expected = Complex64::new(-1.5, 0.625);
        assert!(a.approx_eq(expected, ULP128_EPS, ULP128_EPS));
    }

    #[test]
    fn add_assign_matches_add() {
        let mut a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(4.0, -1.5);
        let expected = a + b;
        a += b;
        assert!(a == expected);
    }

    #[test]
    fn sub_assign() {
        let mut a = Complex64::new(5.75, 2.0);
        let b = Complex64::new(1.25, -3.5);
        a -= b;
        let expected = Complex64::new(4.5, 5.5);
        assert!(a.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn sub_assign_matches_sub() {
        let mut a = Complex64::new(2.0,3.0);
        let b = Complex64::new(4.0,-1.5);
        let expected = a - b;
        a -= b;
        assert!(a == expected);
    }

    #[test]
    fn mul_assign() {
        let mut a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(4.0, -1.5);
        a *= b;
        let expected = Complex64::new(12.5, 9.0);
        assert!(a.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn mul_assign_matches_mul() {
        let mut a = Complex64::new(2.0,3.0);
        let b = Complex64::new(4.0,-1.5);
        let expected = a * b;
        a *= b;
        assert!(a == expected);
    }

    #[test]
    fn div_assign() {
        let mut a = Complex64::new(7.5, 2.5);
        let b = Complex64::new(1.25, -0.75);
        a /= b;
        let expected = Complex64::new(3.5294117647, 4.1176470588);
        assert!(a.approx_eq(expected, 1.0e-9, 1.0e-12));
    }

    #[test]
    fn div_assign_matches_div() {
        let mut a = Complex64::new(2.0,3.0);
        let b = Complex64::new(4.0,-1.5);
        let expected = a / b;
        a /= b;
        assert!(a == expected);
    }

    #[test]
    fn fma() {
        let z = Complex64::new(1.75, -2.25);
        let w = Complex64::new(-0.5, 3.125);
        let a = Complex64::new(4.0, -1.5);
        let fma_result = z.fma(w, a);
        assert!(fma_result.approx_eq(z * w + a, 1e-12, 1e-12));
    }

    #[test]
    fn fma_rounding_sensitive() {
        let z = Complex64::new(1.0e16, 1.0);
        let w = Complex64::new(1.0e-16, -1.0);
        let a = Complex64::new(1.0, -1.0);
        let fma_result = z.fma(w, a);
        assert!(fma_result.approx_eq(z * w + a, 1e-12, 1e-12));
    }

    #[test]
    fn fma_vs_mul_add_stability() {
        let z = Complex64::new(1e16, 1.0);
        let w = Complex64::new(1e-16, -1.0);
        let a = Complex64::new(1.0, -1.0);

        let fma = z.fma(w, a);
        let naive = z * w + a;

        assert!(fma.norm() <= naive.norm() || fma.approx_eq(naive, ULP2_EPS, ULP2_EPS));
    }

    #[test]
    fn mul_by_scalar() {
        let a = Complex64::new(2.0, 2.0);
        let b = a * 2.0;
        let expected = Complex64::new(4.0, 4.0);
        assert!(b.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn mul_scalar_by_complex() {
        let a = Complex64::new(2.0, 2.0);
        let b = 2.0 * a;
        let expected = Complex64::new(4.0, 4.0);
        assert!(b.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn div_by_scalar() {
        let a = Complex64::new(2.0, 2.0);
        let b = a / 2.0;
        let expected = Complex64::new(1.0, 1.0);
        assert!(b.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn abs() {
        let a = Complex64::new(4.23, 2.28);
        let result = a.abs();
        let expected = 4.805340778758568;
        assert!(is_approx_equal(result, expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn conj() {
       let a = Complex64::new(2.0, 5.0);
       assert!(a.conj() == Complex64::new(2.0, -5.0))
    }

    #[test]
    fn sqrt() {
        let a = Complex64::new(3.0, 4.0);
        let w = a.sqrt();
        assert!(w == Complex64::new(2.0, 1.0))
    }
    
    #[test]
    fn sqrt_zero() {
        assert!(Complex64::new(0.0, 0.0).sqrt() == Complex64::new(0.0, 0.0));
    }

    #[test]
    fn sqrt_conjugate() {
        let z = Complex64::new(3.0, 4.0);
        assert!(z.conj().sqrt().approx_eq(z.sqrt().conj(), ULP2_EPS, ULP2_EPS));
    }

    #[test]
    fn sqrt_negative_real_axis() {
        let a = Complex64::new(-4.0, 0.0);
        let w = a.sqrt();
        // principal square root: 0 + 2i
        let expected = Complex64::new(0.0, 2.0);
        assert!(w.approx_eq(expected, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn sqrt_large_magnitude() {
        let a = Complex64::new(1.0e300, 1.0);
        let w = a.sqrt();
        // Property-based check:
        // sqrt(a) * sqrt(a) ≈ z
        let reconstructed = w * w;
        assert!(reconstructed.approx_eq(a, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn sqrt_near_branch_cut() {
        let a = Complex64::new(-1.0, 1.0e-14);
        let w = a.sqrt();
        let reconstructed = w * w;
        assert!(reconstructed.approx_eq(a, ULP128_EPS, ULP2_EPS));
    }

    #[test]
    fn sin() {
        assert!(_0_0i.sin().approx_eq(_0_0i, ULP128_EPS, ULP2_EPS));
        assert!(
            _1_0i.scale(std::f64::consts::PI * 2.0).sin().approx_eq(_0_0i, ULP128_EPS, ULP2_EPS)
        );
        assert!(
            _0_1i.sin().approx_eq(_0_1i.scale(1.0_f64.sinh()), ULP128_EPS, ULP2_EPS)
        );
        for &c in ALL_CONSTS.iter() {
            // sin(conj(z)) = conj(sin(z))
            
            assert!(
                c.conj().sin().approx_eq(c.sin().conj(), ULP128_EPS, ULP2_EPS)
            );
            
            // sin(-z) = -sin(z)
            assert!(
                c.scale(-1.0).sin().approx_eq(c.sin().scale(-1.0), ULP128_EPS, ULP2_EPS)
            );
        }
    }

    #[test]
    fn cos() {
        assert!(_0_0i.cos().approx_eq(_1_0i, ULP128_EPS, ULP2_EPS));
        assert!(
            _1_0i.scale(std::f64::consts::PI * 2.0).cos().approx_eq(_1_0i, ULP128_EPS, ULP2_EPS)
        );
        
        assert!(
            _0_1i.cos().approx_eq(_1_0i.scale(1.0_f64.cosh()), ULP128_EPS, ULP2_EPS)
        );

        for &c in ALL_CONSTS.iter() {
            // cos(conj(z)) = conj(cos(z))
            assert!(
                c.conj().cos().approx_eq(c.cos().conj(), ULP128_EPS, ULP2_EPS)
            );
            
            // cos(-z) = cos(z)
            assert!(
                c.scale(-1.0).cos().approx_eq(c.cos(), ULP128_EPS, ULP2_EPS)
            );
        }
    }

    #[test]
    fn trig_identity() {
        let z = Complex64::new(0.7, 1.3);
        let lhs = z.sin()*z.sin() + z.cos()*z.cos();
        assert!(lhs.approx_eq(Complex64::new(1.0, 0.0), ULP2_EPS, ULP2_EPS));
    }

    #[test]
    fn re_unit() {
        let a = Complex64::re_unit();
        assert!(a == Complex64::new(1.0, 0.0))
    }

    #[test]
    fn im_unit() {
        let a = Complex64::im_unit();
        assert!(a == Complex64::new(0.0, 1.0))
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

    #[test]
    fn norm() {
        let z = Complex64::new(3.0, 4.0);
        assert_eq!(z.norm(), 25.0);
    }

    #[test]
    fn is_zero() {
        assert!(Complex64::new(0.0, 0.0).is_zero());
        assert!(!Complex64::new(1e-300, 0.0).is_zero());
    }

    #[test]
    fn signed_zero() {
        assert!(Complex64::new(-0.0, -0.0).is_zero());
    }

    #[test]
    fn negation() {
        let mut c = Complex64::new(2.0, 4.0);
        c = -c;
        assert!(c.re == -2.0);
        assert!(c.im == -4.0);
    }

    #[test]
    #[should_panic]
    fn div_by_zero() {
        let a = Complex64::new(1.0, 2.0);
        let z = Complex64::new(0.0, 0.0);
        let _ = a / z;
    }

    #[test]
    fn nan_propagation_mul() {
        let a = Complex64::new(f64::NAN, 1.0);
        let b = Complex64::new(1.0, 2.0);
        assert!((a * b).is_nan());
    }
}
