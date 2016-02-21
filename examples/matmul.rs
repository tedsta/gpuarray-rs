extern crate gpuarray as ga;

use ga::Context;
use ga::tensor::{Tensor, TensorMode};
use ga::array::Array;

fn main() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 10], (0..5*10).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![10, 15], (0..10*15).map(|x| (x as f32)*2.0).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_cl: Tensor<f32> = Tensor::new(ctx, vec![5, 15], TensorMode::Mut);

    ga::matmul(ctx, &a_cl, &b_cl, &c_cl);
    
    let c = c_cl.get(ctx);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("A*B = \n{:?}", c);
}
