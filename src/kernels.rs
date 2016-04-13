use std::any::TypeId;
use std::collections::HashMap;

use opencl::hl::{Kernel, Program};

use num::Num;

macro_rules! kernels_hashmap {
    ( $program:ident, $kernel_name:expr, $( $t:ty ),* ) => {
        {
            let mut kernels = HashMap::new();
            $(
                let type_name = stringify!($t);
                kernels.insert(TypeId::of::<$t>(),
                               $program.create_kernel(format!("array_{}_{}",
                                                              $kernel_name, type_name).as_ref()));
            )*
            kernels
        }
    };
}

pub struct Kernels {
    copy_to: HashMap<TypeId, Kernel>,
    fill: HashMap<TypeId, Kernel>,
    sum: HashMap<TypeId, Kernel>,
    add: HashMap<TypeId, Kernel>,
    add_slice: HashMap<TypeId, Kernel>,
    sub: HashMap<TypeId, Kernel>,
    multiply: HashMap<TypeId, Kernel>,
    transpose: HashMap<TypeId, Kernel>,
    matmul: HashMap<TypeId, Kernel>,
    max: HashMap<TypeId, Kernel>,
    dmax: HashMap<TypeId, Kernel>,
    min: HashMap<TypeId, Kernel>,
    dmin: HashMap<TypeId, Kernel>,
    mse: HashMap<TypeId, Kernel>,
    dmse: HashMap<TypeId, Kernel>,
    tanh: HashMap<TypeId, Kernel>,
    dtanh: HashMap<TypeId, Kernel>,
    sigmoid: HashMap<TypeId, Kernel>,
    dsigmoid: HashMap<TypeId, Kernel>,
}

impl Kernels {
    pub fn new(program: &Program) -> Kernels {
        Kernels {
            copy_to: kernels_hashmap!(program, "copy_to", f32, i32),
            fill: kernels_hashmap!(program, "fill", f32, i32),
            sum: kernels_hashmap!(program, "sum", f32, i32),
            add: kernels_hashmap!(program, "add", f32, i32),
            add_slice: kernels_hashmap!(program, "add_slice", i32),
            sub: kernels_hashmap!(program, "sub", f32),
            multiply: kernels_hashmap!(program, "multiply", f32, i32),
            transpose: kernels_hashmap!(program, "transpose", f32, i32),
            matmul: kernels_hashmap!(program, "matmul", f32, i32),
            max: kernels_hashmap!(program, "max", f32),
            dmax: kernels_hashmap!(program, "dmax", f32),
            min: kernels_hashmap!(program, "min", f32),
            dmin: kernels_hashmap!(program, "dmin", f32),
            mse: kernels_hashmap!(program, "mse", f32),
            dmse: kernels_hashmap!(program, "dmse", f32),
            tanh: kernels_hashmap!(program, "tanh", f32),
            dtanh: kernels_hashmap!(program, "dtanh", f32),
            sigmoid: kernels_hashmap!(program, "sigmoid", f32),
            dsigmoid: kernels_hashmap!(program, "dsigmoid", f32),
        }
    }

    pub fn copy_to<T: Num>(&self) -> &Kernel {
        &self.copy_to[&TypeId::of::<T>()]
    }

    pub fn fill<T: Num>(&self) -> &Kernel {
        &self.fill[&TypeId::of::<T>()]
    }

    pub fn sum<T: Num>(&self) -> &Kernel {
        &self.sum[&TypeId::of::<T>()]
    }
    
    pub fn add<T: Num>(&self) -> &Kernel {
        &self.add[&TypeId::of::<T>()]
    }

    pub fn add_slice<T: Num>(&self) -> &Kernel {
        &self.add_slice[&TypeId::of::<T>()]
    }

    pub fn sub<T: Num>(&self) -> &Kernel {
        &self.sub[&TypeId::of::<T>()]
    }

    pub fn multiply<T: Num>(&self) -> &Kernel {
        &self.multiply[&TypeId::of::<T>()]
    }

    pub fn transpose<T: Num>(&self) -> &Kernel {
        &self.transpose[&TypeId::of::<T>()]
    }

    pub fn matmul<T: Num>(&self) -> &Kernel {
        &self.matmul[&TypeId::of::<T>()]
    }

    pub fn max<T: Num>(&self) -> &Kernel {
        &self.max[&TypeId::of::<T>()]
    }

    pub fn dmax<T: Num>(&self) -> &Kernel {
        &self.dmax[&TypeId::of::<T>()]
    }

    pub fn min<T: Num>(&self) -> &Kernel {
        &self.min[&TypeId::of::<T>()]
    }

    pub fn dmin<T: Num>(&self) -> &Kernel {
        &self.dmin[&TypeId::of::<T>()]
    }

    pub fn mse<T: Num>(&self) -> &Kernel {
        &self.mse[&TypeId::of::<T>()]
    }

    pub fn dmse<T: Num>(&self) -> &Kernel {
        &self.dmse[&TypeId::of::<T>()]
    }

    pub fn tanh<T: Num>(&self) -> &Kernel {
        &self.tanh[&TypeId::of::<T>()]
    }

    pub fn dtanh<T: Num>(&self) -> &Kernel {
        &self.dtanh[&TypeId::of::<T>()]
    }

    pub fn sigmoid<T: Num>(&self) -> &Kernel {
        &self.sigmoid[&TypeId::of::<T>()]
    }

    pub fn dsigmoid<T: Num>(&self) -> &Kernel {
        &self.dsigmoid[&TypeId::of::<T>()]
    }
}