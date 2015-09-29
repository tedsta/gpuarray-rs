use opencl::hl::KernelArg;

pub trait Num: Copy { }

impl Num for f32 {}
impl Num for usize {}
