use std::fmt;

use opencl::hl::KernelArg;

pub trait Num: Copy+fmt::Display {
    fn name() -> &'static str;
}

impl Num for f32 { fn name() -> &'static str { "f32" } }
impl Num for f64 { fn name() -> &'static str { "f32" } }
impl Num for i8 { fn name() -> &'static str { "i8" } }
impl Num for i16 { fn name() -> &'static str { "i16" } }
impl Num for i32 { fn name() -> &'static str { "i32" } }
impl Num for i64 { fn name() -> &'static str { "i64" } }
impl Num for u8 { fn name() -> &'static str { "u8" } }
impl Num for u16 { fn name() -> &'static str { "u16" } }
impl Num for u32 { fn name() -> &'static str { "u32" } }
impl Num for u64 { fn name() -> &'static str { "u64" } }
