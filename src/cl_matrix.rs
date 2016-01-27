use std::cell::{RefCell, Ref};

use opencl;
use opencl::mem::{Buffer, CLBuffer};

use context::Context;
use matrix::Matrix;
use num::Num;

pub enum ClMatrixMode {
    In,
    Out,
    Mut,
}

pub struct ClMatrix<T: Num> {
    rows: usize,
    columns: usize,
    buffer: CLBuffer<T>,
    event: RefCell<Option<Event>>,
}

impl<T: Num> ClMatrix<T> {
    pub fn new(ctx: &Context, rows: usize, columns: usize, mode: ClMatrixMode) -> ClMatrix<T> {
        let cl_mem_mode =
            match mode {
                ClMatrixMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                ClMatrixMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                ClMatrixMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        ClMatrix {
            rows: rows,
            columns: columns,
            buffer: ctx.ctx.create_buffer(rows*columns, cl_mem_mode),
            event: RefCell::new(None),
        }
    }

    pub fn from_matrix(ctx: &Context,
                       matrix: &Matrix<T>,
                       mode: ClMatrixMode) -> ClMatrix<T> {
        let cl_mem_mode =
            match mode {
                ClMatrixMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                ClMatrixMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                ClMatrixMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        let cl_matrix =
            ClMatrix {
                rows: matrix.rows(),
                columns: matrix.columns(),
                buffer: ctx.ctx.create_buffer(matrix.rows()*matrix.columns(), cl_mem_mode),
                event: RefCell::new(None),
            };
        
        ctx.queue.write(&cl_matrix.buffer, &&matrix.buffer()[..], ());
        cl_matrix
    }

    pub fn get(&self, ctx: &Context) -> Matrix<T> {
        let vec = ctx.queue.get(&self.buffer, self.get_event().as_ref().map(|x| &**x));
        Matrix::from_vec(self.rows, self.columns, vec)
    }
    
    pub fn set(&self, ctx: &Context, matrix: &Matrix<T>) {
        ctx.queue.write(&self.buffer, &&matrix.buffer()[..], ());
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
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

    pub fn copy_to(&self, ctx: &Context, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_copy_to_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn sum(&self, ctx: &Context, axis: usize, other: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_sum_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        println!("waa");
        kernel.set_arg(2, &self.rows());
        println!("foo");
        kernel.set_arg(3, &self.columns());
        println!("bar");
        kernel.set_arg(4, &axis);

        let axis_dim = [self.rows(), self.columns()][axis];

        let new_event = {
            ctx.queue.enqueue_async_kernel(&kernel, axis_dim, None, self.get_event().as_ref().map(|x| &**x))
        };
        other.set_event(new_event);
    }

    pub fn add(&self, ctx: &Context, axis: i32, other: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_add_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.columns());
        kernel.set_arg(4, &axis);


        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, (self.rows(), self.columns()), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn sub(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_sub_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn multiply(&self, ctx: &Context, axis: i32, other: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_multiply_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.columns());
        kernel.set_arg(4, &axis);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel, (self.rows(), self.columns()), None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn transpose(&self, ctx: &Context, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_transpose_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &self.rows);
        kernel.set_arg(3, &self.columns);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, (self.rows, self.columns),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn dot(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dot_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.columns);
        kernel.set_arg(4, &other.columns);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), other.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           (self.rows, other.columns),
                                           None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn max(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_max_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn min(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_min_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }


    pub fn dmax(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dmax_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn dmin(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dmin_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.set_event(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                        None, self.get_event().as_ref().map(|x| &**x)));
    }

    pub fn mse(&self, ctx: &Context, train: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_mse_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &train.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.rows);
        kernel.set_arg(4, &self.columns);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), train.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           self.columns,
                                           None, event_list)
        };
        output.set_event(new_event);
    }

    pub fn dmse(&self, ctx: &Context, train: &ClMatrix<T>, output: &ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dmse_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &train.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.rows);
        kernel.set_arg(4, &self.columns);

        let new_event = {
            let event_list: &[Option<Ref<Event>>] = &[self.get_event(), train.get_event()];
            ctx.queue.enqueue_async_kernel(&kernel,
                                           (self.rows, self.columns),
                                           None, event_list)
        };
        output.set_event(new_event);
    }
}

pub type Event = opencl::hl::Event;

#[test]
fn cl_matrix_sum_axis() {
    let ref ctx = Context::new();

    let a: Matrix<i32> = Matrix::from_vec(5, 3, (0..15).collect());
    let b = Matrix::from_vec(1, 3, (0..3).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::Out);

    a_cl.sum(ctx, 0, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[30, 35, 40]);
}

#[test]
fn cl_matrix_add() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(1, 10000, (0..10000).collect());
    let b = Matrix::from_vec(1, 10000, (0..10000).map(|x| x*2).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix<i32> = ClMatrix::new(ctx, 1, 10000, ClMatrixMode::Out);

    a_cl.add(ctx, -1, &b_cl, &c_cl);
    
    let c = c_cl.get(ctx);

    for i in 0..10000 {
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}

#[test]
fn cl_matrix_add_reuse() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(1, 10000, (0..10000).collect());
    let b = Matrix::from_vec(1, 10000, (0..10000).map(|x| x*2).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);

    a_cl.add(ctx, -1, &b_cl, &b_cl); // b = a+b
}

#[test]
fn cl_matrix_add_axis() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(5, 3, (0..15).collect());
    let b = Matrix::from_vec(1, 3, (0..3).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);

    a_cl.add(ctx, 0, &b_cl, &a_cl); // a = a+b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0, 2, 4,
                            3, 5, 7,
                            6, 8, 10,
                            9, 11, 13,
                            12, 14, 16]);
}

#[test]
fn cl_matrix_multiply_axis1() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(3, 5, (0..15).collect());
    let b = Matrix::from_vec(3, 1, (0..3).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);

    a_cl.multiply(ctx, 1, &b_cl, &a_cl); // a = a*b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0,  0,  0,  0,  0,
                            5,  6,  7,  8,  9,
                            20, 22, 24, 26, 28]);
}
