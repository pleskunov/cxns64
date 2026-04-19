# 64-bit Floating Complex Numbers

A lightweight Rust library implementing a 64-bit complex number type with C ABI compatibility.

The goal of this library is not to be a Swiss army knife like many similar projects. 
Instead, it focuses on providing a simple, numerically robust, and stable solution with a frozen API. 
It supports only 64-bit floating-point numbers (`f64`) and nothing else.

This is a fundamental building block, and libraries at this level should stay small, predictable, and easy to audit. 
That means no unnecessary abstractions and no dependency chains that pull in half the ecosystem.
The code is intentionally minimal: easy to read, easy to audit. 

No bloat. Just the Rust standard library.

## Installation

After looking into the process of publishing crates and the usual naming disputes around them, 
I decided not to deal with that mess. This library is meant to be used directly, and installation takes less than 30 seconds:

1. Clone the repository next to your project directory.

2. Add this to your `Cargo.toml`:

```toml
[dependencies]
cxns64 = { path = "../cxns64" }
```

3. Import it with `use cxns64::Complex64 as c64;` and you're good to go.

## Compatibility

The `cxns64` crate is tested with rustc 1.94 and newer.
