use std::cell::{RefCell, Ref};

use opencl;
use opencl::hl::KernelArg;
use opencl::mem::{Buffer, CLBuffer};
use libc;
use ref_filter_map::ref_filter_map;

use array::Array;
use context::Context;
use helper;
use num::Num;
use range_arg::RangeArg;

pub enum TensorMode {
    In,
    Out,
    Mut,
}

pub struct Tensor<T: Num> {
    shape: Vec<usize>,
    dim_steps: Vec<usize>,
    buffer: CLBuffer<T>,
    event: RefCell<Option<Event>>,
}

impl<T: Num> Tensor<T> {
    pub fn new(ctx: &Context, shape: Vec<usize>, mode: TensorMode) -> Tensor<T> {
        let mem_mode =
            match mode {
                TensorMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                TensorMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                TensorMode::Mut => { opencl::cl::CL_MEM_READ_WRITE },
            };
        let buf_size = shape.iter().fold(1, |a, b| a*b);
        let dim_steps = helper::compute_dim_steps(&shape);
        Tensor {
            shape: shape,
            dim_steps: dim_steps,
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
            dim_steps: array.dim_steps().to_owned(),
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
            ref_filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

    pub fn slice<'t, 'r>(&'t self, r: &'r [RangeArg]) -> TensorView<'t, 'r, T> {
        TensorView {
            shape: self.shape.as_ref(),
            dim_steps: self.dim_steps.as_ref(),
            ranges: r,
            buffer: &self.buffer,
            event: &self.event,
        }
    }
}

impl<T: Num> KernelArg for Tensor<T> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        self.buffer.get_value()
    }
}

impl<'t, 'r, T: Num> KernelArg for TensorView<'t, 'r, T> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        self.buffer.get_value()
    }
}

pub struct TensorView<'t, 'r, T: Num+'t> {
    pub shape: &'t [usize],
    pub dim_steps: &'t [usize],
    ranges: &'r [RangeArg],
    buffer: &'t CLBuffer<T>,
    event: &'t RefCell<Option<Event>>,
}

impl<'t, 'r, T: Num> TensorView<'t, 'r, T> {
    pub fn set_event(&self, e: Event) {
        *self.event.borrow_mut() = Some(e);
    }

    pub fn get_event(&self) -> Option<Ref<Event>> {
        if self.event.borrow().is_some() {
            ref_filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

    pub fn view_offset(&self, dim: usize) -> usize {
        self.ranges[dim].start
    }

    pub fn view_shape(&self, dim: usize) -> usize {
        self.ranges[dim].len(self.shape[dim])
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub type Event = opencl::hl::Event;

////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_tensor_slicing() {
    use ops::add_slice;

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![4, 3], vec![2, 3, 4,
                                             6, 7, 8,
                                             10, 11, 12,
                                             14, 15, 16]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(1..3), RangeArg::from(1)];
    let atv = at.slice(slice);

    let b = Array::from_vec(vec![4, 4], vec![1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16]);
    let bt = Tensor::from_array(ctx, &b, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(1..3), RangeArg::from(3)];
    let btv = bt.slice(slice);

    let c = Array::from_vec(vec![4, 4], vec![0; 16]);
    let ct = Tensor::from_array(ctx, &c, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(2..4), RangeArg::from(0)];
    let ctv = ct.slice(slice);

    add_slice(ctx, &atv, &btv, &ctv);
    assert!(ct.get(ctx).buffer() == &[0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      15, 0, 0, 0,
                                      23, 0, 0, 0]);
}
