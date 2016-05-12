#[macro_use] extern crate gpuarray as ga;

use ga::{Array, Context, Tensor, TensorMode, add_slice};

fn main() {

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![4, 3], vec![2, 3, 4,
                                             6, 7, 8,
                                             10, 11, 12,
                                             14, 15, 16]);
    let a_gpu = Tensor::from_array(ctx, &a, TensorMode::Mut);

    let b = Array::from_vec(vec![4, 4], vec![1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16]);
    let b_gpu = Tensor::from_array(ctx, &b, TensorMode::Mut);

    let c = Array::from_vec(vec![4, 4], vec![0; 16]);
    let c_gpu = Tensor::from_array(ctx, &c, TensorMode::Mut);

    add_slice(ctx, &a_gpu.slice(s![1..3, 1]), &b_gpu.slice(s![1..3, 3]), &c_gpu.slice(s![2..4, 0]));
    //println!("{:?}", ct.get(ctx));
    assert!(c_gpu.get(ctx).buffer() == &[0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         15, 0, 0, 0,
                                         23, 0, 0, 0]);
}
