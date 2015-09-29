use opencl;
use opencl::mem::{Buffer, CLBuffer};

use context::Context;
use matrix2::Matrix2;
use num::Num;

pub enum ClMatrixMode {
    In,
    Out,
    Mut,
}

pub struct ClMatrix2<T: Num> {
    rows: usize,
    columns: usize,
    buffer: CLBuffer<T>,
}

impl<T: Num> ClMatrix2<T> {
    pub fn new(ctx: &Context, rows: usize, columns: usize, mode: ClMatrixMode) -> ClMatrix2<T> {
        let cl_mem_mode =
            match mode {
                ClMatrixMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                ClMatrixMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                ClMatrixMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        ClMatrix2 {
            rows: rows,
            columns: columns,
            buffer: ctx.ctx.create_buffer(rows*columns, cl_mem_mode),
        }
    }

    pub fn from_matrix(ctx: &Context,
                       matrix: &Matrix2<T>,
                       mode: ClMatrixMode) -> ClMatrix2<T> {
        let cl_mem_mode =
            match mode {
                ClMatrixMode::In => { opencl::cl::CL_MEM_READ_ONLY },
                ClMatrixMode::Out => { opencl::cl::CL_MEM_WRITE_ONLY },
                ClMatrixMode::Mut => { opencl::cl::CL_MEM_READ_WRITE},
            };
        let cl_matrix =
            ClMatrix2 {
                rows: matrix.rows(),
                columns: matrix.columns(),
                buffer: ctx.ctx.create_buffer(matrix.rows()*matrix.columns(), cl_mem_mode),
            };
        
        ctx.queue.write(&cl_matrix.buffer, &&matrix.buffer()[..], ());
        cl_matrix
    }

    pub fn add(&self, ctx: &Context, other: &ClMatrix2<T>, output: &ClMatrix2<T>) -> Event {
        let kernel = ctx.program.create_kernel("vector_add");

        kernel.set_arg(0, &self.buffer);
        kernel.set_arg(1, &other.buffer);
        kernel.set_arg(2, &output.buffer);

        let event = ctx.queue.enqueue_async_kernel(&kernel, self.buffer.len(), None, ());
        Event(event)
    }
}

pub struct Event(opencl::hl::Event);

impl Event {
    pub fn get<T: Num>(&self, ctx: &Context, cl_matrix: &ClMatrix2<T>) -> Matrix2<T> {
        let vec = ctx.queue.get(&cl_matrix.buffer, &self.0);
        Matrix2::from_vec(cl_matrix.rows, cl_matrix.columns, vec)
    }
}

#[test]
fn cl_matrix_add() {
    let ref ctx = Context::new();

    let a = Matrix2::from_vec(1, 10000, (0..10000).collect());
    let b = Matrix2::from_vec(1, 10000, (0..10000).map(|x| x*2).collect());

    let a_cl = ClMatrix2::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix2::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix2<usize> = ClMatrix2::new(ctx, 1, 10000, ClMatrixMode::Out);

    let event = a_cl.add(ctx, &b_cl, &c_cl);
    
    let c = event.get(ctx, &c_cl);

    for i in 0..10000 {
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}
