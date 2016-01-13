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
    let c_cl: ClMatrix<f32> = ClMatrix::new(ctx, 100, 1000, ClMatrixMode::Mut);
    let d_cl: ClMatrix<f32> = ClMatrix::new(ctx, 1000, 100, ClMatrixMode::Out);

    let c_event = a_cl.add(ctx, &b_cl, &c_cl, &[]);
    let d_event = c_cl.transpose(ctx, &d_cl, &[c_event]);
    
    let d = d_event.get(ctx, &d_cl);

    for i in 0..100 {
        for j in 0..1000 {
            assert!(d[(j, i)] == a[(i, j)] + b[(i, j)]);
        }
    }
}
