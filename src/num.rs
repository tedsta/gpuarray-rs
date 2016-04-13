use std::fmt;
use std::marker::Reflect;

use opencl::hl::KernelArg;

pub trait Num: KernelArg+Copy+fmt::Display+Reflect+'static { }

impl Num for f32 { }
//impl Num for f64 { }
//impl Num for i8 { }
//impl Num for i16 { }
impl Num for i32 { }
impl Num for i64 { }
//impl Num for u8 { }
//impl Num for u16 { }
impl Num for u32 { }
impl Num for u64 { }
