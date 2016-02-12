use std::cell::{RefCell, Ref};

use opencl;
use opencl::mem::{Buffer, CLBuffer};

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
                TensorMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        let tensor =
            Tensor {
                shape: array.shape().to_vec(),
                buffer: ctx.ctx.create_buffer(array.buffer().len(), mem_mode),
                event: RefCell::new(None),
            };
        
        ctx.queue.write(&tensor.buffer, &&array.buffer()[..], ());
        tensor
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
    
    fn set_event(&self, e: Event) {
        *self.event.borrow_mut() = Some(e);
    }

    fn get_event(&self) -> Option<Ref<Event>> {
        if self.event.borrow().is_some() {
            Ref::filter_map(self.event.borrow(), |o| o.as_ref())
        } else {
            None
        }
    }

    pub fn copy_to(&self, ctx: &Context, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_copy_to_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn sum(&self, ctx: &Context, axis: usize, other: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_sum_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &self.shape[0]);
        kernel.set_arg(3, &self.shape[1]);
        kernel.set_arg(4, &axis);

        let keep_dim = [self.shape[1], self.shape[0]][axis];

        let new_event = {
            ctx.queue.enqueue_async_kernel(&kernel, keep_dim, None, self.get_event().as_ref().map(|x| &**x))
        };
        other.set_event(new_event);
    }

    pub fn add(&self, ctx: &Context, axis: i32, other: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_add_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.shape[1]);
        kernel.set_arg(4, &axis);


        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, (self.shape[0], self.shape[1]), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn sub(&self, ctx: &Context, other: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_sub_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn multiply(&self, ctx: &Context, axis: i32, other: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_multiply_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.shape[1]);
        kernel.set_arg(4, &axis);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, (self.shape[0], self.shape[1]), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn transpose(&self, ctx: &Context, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_transpose_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &self.shape[0]);
        kernel.set_arg(3, &self.shape[1]);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, (self.shape[0], self.shape[1]),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn dot(&self, ctx: &Context, other: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_dot_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.shape[1]);
        kernel.set_arg(4, &other.shape[1]);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           (self.shape[0], other.shape[1]),
                                           None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn max(&self, ctx: &Context, threshold: T, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_max_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn min(&self, ctx: &Context, threshold: T, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_min_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }


    pub fn dmax(&self, ctx: &Context, threshold: T, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_dmax_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn dmin(&self, ctx: &Context, threshold: T, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_dmin_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn mse(&self, ctx: &Context, train: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_mse_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &train.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.shape[0]);
        kernel.set_arg(4, &self.shape[1]);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), train.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           self.shape[1],
                                           None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn dmse(&self, ctx: &Context, train: &Tensor<T>, output: &Tensor<T>) {
        let kernel = ctx.program.create_kernel(format!("array_dmse_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &train.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.shape[0]);
        kernel.set_arg(4, &self.shape[1]);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), train.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           (self.shape[0], self.shape[1]),
                                           None, event_list)
        };
        output.set_event(new_event);
    }
}

pub type Event = opencl::hl::Event;

#[test]
fn tensor_sum_axis0() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![1, 3], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::Out);

    a_cl.sum(ctx, 0, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[30, 35, 40]);
}

#[test]
fn tensor_sum_axis1() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![5, 1], (0..5).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::Out);

    a_cl.sum(ctx, 1, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[3, 12, 21, 30, 39]);
}

#[test]
fn tensor_add() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 10000], (0..5*10000).collect());
    let b = Array::from_vec(vec![5, 10000], (0..5*10000).map(|x| x*2).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_cl: Tensor<i32> = Tensor::new(ctx, vec![5, 10000], TensorMode::Out);

    a_cl.add(ctx, -1, &b_cl, &c_cl);
    
    let c = c_cl.get(ctx);

    for i in 0..5 {
        for j in 0..10000 {
            assert!(c[&[i, j]] == a[&[i, j]] + b[&[i, j]]);
        }
    }
}

#[test]
fn tensor_add_reuse() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![1, 10000], (0..10000).collect());
    let b = Array::from_vec(vec![1, 10000], (0..10000).map(|x| x*2).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    a_cl.add(ctx, -1, &b_cl, &b_cl); // b = a+b
}

#[test]
fn tensor_add_axis() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![1, 3], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    a_cl.add(ctx, 0, &b_cl, &a_cl); // a = a+b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0, 2, 4,
                            3, 5, 7,
                            6, 8, 10,
                            9, 11, 13,
                            12, 14, 16]);
}

#[test]
fn tensor_multiply_axis1() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![3, 5], (0..15).collect());
    let b = Array::from_vec(vec![3, 1], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    a_cl.multiply(ctx, 1, &b_cl, &a_cl); // a = a*b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0,  0,  0,  0,  0,
                            5,  6,  7,  8,  9,
                            20, 22, 24, 26, 28]);
}
