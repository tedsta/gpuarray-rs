#![feature(convert)]

extern crate opencl;

pub use context::Context;

pub mod context;
pub mod cl_matrix;
pub mod matrix;
pub mod num;
