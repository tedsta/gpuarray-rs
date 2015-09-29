extern crate matrix;

use matrix::Context;
use matrix::cl_matrix2::{ClMatrix2, ClMatrixMode};
use matrix::matrix2::Matrix2;

fn main() {
    let ref ctx = Context::new();

    let a = Matrix2::from_vec(10000, 1000, (0..10000*1000).collect());
    let b = Matrix2::from_vec(10000, 1000, (0..10000*1000).map(|x| x*2).collect());

    let a_cl = ClMatrix2::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix2::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix2<usize> = ClMatrix2::new(ctx, 1000, 10000, ClMatrixMode::Out);

    let event = a_cl.add(ctx, &b_cl, &c_cl);
    
    let c = event.get(ctx, &c_cl);

    for i in 0..1000*10000{
        assert!(c[(0, i)] == a[(0, i)] + b[(0, i)]);
    }
}
