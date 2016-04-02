#![feature(iter_arith)]

extern crate opencl;
extern crate libc;
extern crate ref_filter_map;

pub use context::Context;
pub use array::Array;
pub use tensor::{Event, Tensor, TensorMode};
pub use ops::*;
pub use range_arg::RangeArg;

pub mod context;
pub mod tensor;
pub mod array;
pub mod num;
pub mod ops;
pub mod range_arg;

mod helper;
