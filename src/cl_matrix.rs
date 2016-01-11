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
            };
        
        ctx.queue.write(&cl_matrix.buffer, &&matrix.buffer()[..], ());
        cl_matrix
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

    pub fn copy_to(&self, ctx: &Context, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_copy_to_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn add(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_add_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn sub(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_sub_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn dot(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_dot_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn transpose(&self, ctx: &Context, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_transpose_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &self.rows);
        kernel.set_arg(3, &self.columns);

        let event = ctx.queue.enqueue_async_kernel(&kernel,
                                                   (self.rows, self.columns),
                                                   None, ());
        Event(event)
    }

    pub fn cross(&self, ctx: &Context, other: &ClMatrix<T>, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_cross_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);
        kernel.set_arg(3, &self.columns);
        kernel.set_arg(4, &other.columns);

        let event = ctx.queue.enqueue_async_kernel(&kernel,
                                                   (self.rows, other.columns),
                                                   None, ());
        Event(event)
    }

    pub fn max(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_max_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn min(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_min_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }


    pub fn dmax(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_dmax_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }

    pub fn dmin(&self, ctx: &Context, threshold: T, output: &ClMatrix<T>) -> Event {
        let kernel = ctx.program.create_kernel(format!("vector_dmin_{}", T::name()).as_str());

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &output.buffer);
        kernel.set_arg(2, &threshold);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }
}

pub struct Event(opencl::hl::Event);

impl Event {
    pub fn get<T: Num>(&self, ctx: &Context, cl_matrix: &ClMatrix<T>) -> Matrix<T> {
        let vec = ctx.queue.get(&cl_matrix.buffer, &self.0);
        Matrix::from_vec(cl_matrix.rows, cl_matrix.columns, vec)
    }
}

#[test]
fn cl_matrix_add() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(1, 10000, (0..10000).collect());
    let b = Matrix::from_vec(1, 10000, (0..10000).map(|x| x*2).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix<i32> = ClMatrix::new(ctx, 1, 10000, ClMatrixMode::Out);

    let event = a_cl.add(ctx, &b_cl, &c_cl);
    
    let c = event.get(ctx, &c_cl);

    for i in 0..10000 {
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}
