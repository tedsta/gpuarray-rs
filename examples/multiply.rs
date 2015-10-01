extern crate matrix;

use matrix::Context;
use matrix::cl_matrix::{ClMatrix, ClMatrixMode};
use matrix::matrix::Matrix;

fn main() {
    let ref ctx = Context::new();

    let a = Matrix::from_vec(5, 10, (0..5*10).map(|x| x as f32).collect());
    let b = Matrix::from_vec(10, 15, (0..10*15).map(|x| (x as f32)*2.0).collect());

    let a_cl = ClMatrix::from_matrix(ctx, &a, ClMatrixMode::In);
    let b_cl = ClMatrix::from_matrix(ctx, &b, ClMatrixMode::In);
    let c_cl: ClMatrix<f32> = ClMatrix::new(ctx, 5, 15, ClMatrixMode::Mut);

    let event = a_cl.multiply(ctx, &b_cl, &c_cl);
    
    let c = event.get(ctx, &c_cl);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("A*B = \n{:?}", c);
}
