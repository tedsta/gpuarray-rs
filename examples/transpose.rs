extern crate matrix;

use matrix::Context;
use matrix::tensor::{Tensor, TensorMode};
use matrix::array::Array;

fn main() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![100, 1000], (0..100*1000).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![100, 1000], (0..100*1000).map(|x| (x as f32)*2.0).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);
    let mut c_cl: Tensor<f32> = Tensor::new(ctx, vec![100, 1000], TensorMode::Mut);
    let mut d_cl: Tensor<f32> = Tensor::new(ctx, vec![1000, 100], TensorMode::Out);

    a_cl.add(ctx, -1, &b_cl, &mut c_cl);
    c_cl.transpose(ctx, &mut d_cl);
    
    let d = d_cl.get(ctx);

    for i in 0..100 {
        for j in 0..1000 {
            assert!(d[&[j, i]] == a[&[i, j]] + b[&[i, j]]);
        }
    }
}
