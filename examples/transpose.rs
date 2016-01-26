extern crate matrix;

use matrix::Context;
use matrix::cl_matrix::{ClMatrix, ClMatrixMode};
use matrix::matrix::Matrix;

fn main() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(100, 1000, (0..100*1000).map(|x| x as f32).collect());
    let b = Matrix::from_vec(100, 1000, (0..100*1000).map(|x| (x as f32)*2.0).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let mut c_cl: ClMatrix<f32> = ClMatrix::new(ctx, 100, 1000, ClMatrixMode::Mut);
    let mut d_cl: ClMatrix<f32> = ClMatrix::new(ctx, 1000, 100, ClMatrixMode::Out);

    a_cl.add(ctx, -1, &b_cl, &mut c_cl);
    c_cl.transpose(ctx, &mut d_cl);
    
    let d = d_cl.get(ctx);

    for i in 0..100 {
        for j in 0..1000 {
            assert!(d[(j, i)] == a[(i, j)] + b[(i, j)]);
        }
    }
}
