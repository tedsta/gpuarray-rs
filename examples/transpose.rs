extern crate gpuarray as ga;

use ga::Context;
use ga::tensor::{Tensor, TensorMode};
use ga::array::Array;

fn main() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![100, 1000], (0..100*1000).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![100, 1000], (0..100*1000).map(|x| (x as f32)*2.0).collect());

    let a_gpu = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_gpu = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_gpu = Tensor::new(ctx, vec![100, 1000], TensorMode::In);
    let d_gpu: Tensor<f32> = Tensor::new(ctx, vec![1000, 100], TensorMode::Out);

    ga::add(ctx, &a_gpu, -1, &b_gpu, &c_gpu);
    ga::transpose(ctx, &c_gpu, &d_gpu);
    
    let d = d_gpu.get(ctx);

    for i in 0..100 {
        for j in 0..1000 {
            assert!(d[&[j, i]] == a[&[i, j]] + b[&[i, j]]);
        }
    }
}
