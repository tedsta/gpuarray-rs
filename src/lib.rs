#![feature(convert)]
#![feature(cell_extras)]

extern crate opencl;

pub use context::Context;
pub use matrix::Matrix;
pub use cl_matrix::{Event, ClMatrix, ClMatrixMode};

pub mod context;
pub mod cl_matrix;
pub mod matrix;
pub mod num;
