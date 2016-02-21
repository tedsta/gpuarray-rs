use std::cell::{RefCell, Ref};

use opencl;
use opencl::hl::KernelArg;
use opencl::mem::{Buffer, CLBuffer};
use libc;

use context::Context;
use array::Array;
use num::Num;

pub enum TensorMode {
    In,
    Out,
    Mut,
}

pub struct Tensor<T: Num> {
    shape: Vec<usize>,
    buffer: CLBuffer<T>,
    event: RefCell<Option<Event>>,
}

impl<T: Num> Tensor<T> {
    pub fn new(ctx: &Context, shape: Vec<usize>, mode: TensorMode) -> Tensor<T> {
        let mem_mode =
            match mode {
                TensorMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                TensorMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                TensorMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        let buf_size = shape.iter().fold(1, |a, b| a*b);
        Tensor {
            shape: shape,
            buffer: ctx.ctx.create_buffer(buf_size, mem_mode),
            event: RefCell::new(None),
        }
    }

    pub fn from_array(ctx: &Context,
                      array: &Array<T>,
                      mode: TensorMode) -> Tensor<T> {
        let mem_mode =
            match mode {
                TensorMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                TensorMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                TensorMode::Mut => { opencl::cl::CL_MEM_READ_WRITE },
            };
        Tensor {
            shape: array.shape().to_vec(),
            buffer: ctx.ctx.create_buffer_from(array.buffer(), mem_mode),
            event: RefCell::new(None),
        }
    }

    pub fn get(&self, ctx: &Context) -> Array<T> {
        let vec = ctx.queue.get(&self.buffer, self.get_event().as_ref().map(|x| &**x));
        Array::from_vec(self.shape.clone(), vec)
    }
    
    pub fn set(&self, ctx: &Context, array: &Array<T>) {
        ctx.queue.write(&self.buffer, &&array.buffer()[..], ());
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn set_event(&self, e: Event) {
        *self.event.borrow_mut() = Some(e);
    }

    pub fn get_event(&self) -> Option<Ref<Event>> {
        if self.event.borrow().is_some() {
            Ref::filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

}

impl<T: Num> KernelArg for Tensor<T> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        self.buffer.get_value()
    }
}

pub type Event = opencl::hl::Event;
