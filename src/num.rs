use opencl::hl::KernelArg;

pub trait Num: Copy+KernelArg { }

impl Num for f32 {}
impl Num for usize {}
