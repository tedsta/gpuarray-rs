#![feature(cell_extras)]
#![feature(iter_arith)]

extern crate opencl;
extern crate libc;

pub use context::Context;
pub use array::Array;
pub use tensor::{Event, Tensor, TensorMode};
pub use ops::*;

pub mod context;
pub mod tensor;
pub mod array;
pub mod num;
pub mod ops;
