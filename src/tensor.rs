use std::rc::Rc;
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
    event: RefCell<Option<Rc<Event>>>,
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
        let vec = ctx.queue.get(&self.buffer, self.get_event().as_ref().map(|x| &***x));
        Array::from_vec(self.shape.clone(), vec)
    }

    pub fn read(&self, ctx: &Context, array: &mut Array<T>) {
        ctx.queue.read(&self.buffer, &mut array.buffer_mut(), self.get_event().as_ref().map(|x| &***x));
    }
    
    pub fn set(&self, ctx: &Context, array: &Array<T>) {
        ctx.queue.write(&self.buffer, &array.buffer(), ());
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim_steps(&self) -> &[usize] {
        &self.dim_steps
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn set_event(&self, e: Rc<Event>) {
        *self.event.borrow_mut() = Some(e);
    }

    pub fn get_event(&self) -> Option<Ref<Rc<Event>>> {
        if self.event.borrow().is_some() {
            ref_filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

    pub fn slice<'t, R: AsRef<[RangeArg]>>(&'t self, r: R) -> TensorView<'t, T, R> {
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

impl<'t, T: Num, R: AsRef<[RangeArg]>> KernelArg for TensorView<'t, T, R> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        self.buffer.get_value()
    }
}

pub struct TensorView<'t, T: Num+'t, R: AsRef<[RangeArg]>> {
    pub shape: &'t [usize],
    pub dim_steps: &'t [usize],
    ranges: R,
    buffer: &'t CLBuffer<T>,
    event: &'t RefCell<Option<Rc<Event>>>,
}

impl<'t, T: Num, R: AsRef<[RangeArg]>> TensorView<'t, T, R> {
    pub fn set_event(&self, e: Rc<Event>) {
        *self.event.borrow_mut() = Some(e);
    }

    pub fn get_event(&self) -> Option<Ref<Rc<Event>>> {
        if self.event.borrow().is_some() {
            ref_filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

    pub fn view_offset(&self, dim: usize) -> usize {
        self.ranges.as_ref().get(dim).map(|r| r.start).unwrap_or(0)
    }

    pub fn view_shape(&self, dim: usize) -> usize {
        self.ranges.as_ref().get(dim).map(|r| r.len(self.shape[dim])).unwrap_or(self.shape[dim])
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub type Event = opencl::hl::Event;

////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_tensor_read() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![2, 2], vec![1i32, 2,
                                             3, 4]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Out);
    let mut b = Array::new(vec![2, 2], 0);

    at.read(ctx, &mut b);

    assert!(b.buffer() == &[1, 2,
                            3, 4]);
}

#[test]
fn test_tensor_incomplete_slice() {
    let ref ctx = Context::new();
    let t = Tensor::<f32>::new(ctx, vec![4, 5, 6], TensorMode::Out);
    let t_slice = t.slice(s![1..3]);
    assert!(t_slice.view_offset(0) == 1);
    assert!(t_slice.view_offset(1) == 0);
    assert!(t_slice.view_offset(2) == 0);
    assert!(t_slice.view_shape(0) == 2);
    assert!(t_slice.view_shape(1) == 5);
    assert!(t_slice.view_shape(2) == 6);
}
