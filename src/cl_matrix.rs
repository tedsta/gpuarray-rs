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
    event: Option<Event>,
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
            event: None,
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
                event: None,
            };
        
        ctx.queue.write(&cl_matrix.buffer, &&matrix.buffer()[..], ());
        cl_matrix
    }

    pub fn get(&self, ctx: &Context) -> Matrix<T> {
        let vec = ctx.queue.get(&self.buffer, self.event.as_ref().unwrap());
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

    pub fn copy_to(&self, ctx: &Context, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_copy_to_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, self.event.as_ref()));
    }

    pub fn add(&self, ctx: &Context, other: &ClMatrix<T>, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_add_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event_list: &[Option<&Event>] = &[self.event.as_ref(), other.event.as_ref()];
        output.event = Some(ctx.queue
                               .enqueue_async_kernel(&kernel, self.buffer.len(),
                                                     None, event_list));
    }

    pub fn sub(&self, ctx: &Context, other: &ClMatrix<T>, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_sub_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event_list: &[Option<&Event>] = &[self.event.as_ref(), other.event.as_ref()];
        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, event_list));
    }

    pub fn dot(&self, ctx: &Context, other: &ClMatrix<T>, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dot_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event_list: &[Option<&Event>] = &[self.event.as_ref(), other.event.as_ref()];
        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, event_list));
    }

    pub fn transpose(&self, ctx: &Context, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_transpose_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &self.rows);
        kernel.set_arg(3, &self.columns);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, (self.rows, self.columns),
                                                           None, self.event.as_ref()));
    }

    pub fn cross(&self, ctx: &Context, other: &ClMatrix<T>, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_cross_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.columns);
        kernel.set_arg(4, &other.columns);

        let event_list: &[Option<&Event>] = &[self.event.as_ref(), other.event.as_ref()];
        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel,
                                                           (self.rows, other.columns),
                                                           None, event_list));
    }

    pub fn max(&self, ctx: &Context, threshold: T, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_max_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, self.event.as_ref()));
    }

    pub fn min(&self, ctx: &Context, threshold: T, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_min_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, self.event.as_ref()));
    }


    pub fn dmax(&self, ctx: &Context, threshold: T, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dmax_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, self.event.as_ref()));
    }

    pub fn dmin(&self, ctx: &Context, threshold: T, output: &mut ClMatrix<T>) {
        let kernel = ctx.program.create_kernel(format!("vector_dmin_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        output.event = Some(ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(),
                                                           None, self.event.as_ref()));
    }
}

pub type Event = opencl::hl::Event;

#[test]
fn cl_matrix_add() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(1, 10000, (0..10000).collect());
    let b = Matrix::from_vec(1, 10000, (0..10000).map(|x| x*2).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let mut c_cl: ClMatrix<i32> = ClMatrix::new(ctx, 1, 10000, ClMatrixMode::Out);

    a_cl.add(ctx, &b_cl, &mut c_cl);
    
    let c = c_cl.get(ctx);

    for i in 0..10000 {
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}
