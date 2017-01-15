extern crate gpuarray as ga;

use ga::Context;
use ga::tensor::{Tensor, TensorMode};
use ga::array::Array;

fn main() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 15], (0..5*15).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![15, 10], (0..15*10).map(|x| (x as f32)*2.0).collect());
    let c = Array::from_vec(vec![5, 10], vec![1.0; 5*10]);

    let a_gpu = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_gpu = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_gpu = Tensor::from_array(ctx, &c, TensorMode::In);
    // Our intermediate result must be TensorMode::Mut
    let d_gpu: Tensor<f32> = Tensor::new(ctx, vec![5, 10], TensorMode::Mut);
    let e_gpu: Tensor<f32> = Tensor::new(ctx, vec![5, 10], TensorMode::Out);

    ga::matmul(ctx, &a_gpu, &b_gpu, &d_gpu);
    ga::add(ctx, &d_gpu, -1, &c_gpu, &e_gpu);
    
    let d = d_gpu.get(ctx);
    let e = e_gpu.get(ctx);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("C = \n{:?}", c);
    println!("D = A * B  = \n{:?}", d);
    println!("D + C = \n{:?}", e);
}
