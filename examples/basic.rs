extern crate matrix;

use matrix::Context;
use matrix::cl_matrix::{ClMatrix, ClMatrixMode};
use matrix::matrix::Matrix;

fn main() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(10000, 1000, (0..10000*1000).map(|x| x as f32).collect());
    let b = Matrix::from_vec(10000, 1000, (0..10000*1000).map(|x| (x as f32)*2.0).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix<f32> = ClMatrix::new(ctx, 10000, 1000, ClMatrixMode::Out);

    let event = a_cl.add(ctx, &b_cl, &c_cl);
    
    let c = event.get(ctx, &c_cl);

    for i in 0..10000*1000{
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}
