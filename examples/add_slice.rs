extern crate gpuarray as ga;

use ga::{Array, Context, RangeArg, Tensor, TensorMode, add_slice};

fn main() {

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![4, 3], vec![2, 3, 4,
                                             6, 7, 8,
                                             10, 11, 12,
                                             14, 15, 16]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(1..3), RangeArg::from(1)];
    let atv = at.slice(slice);

    let b = Array::from_vec(vec![4, 4], vec![1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16]);
    let bt = Tensor::from_array(ctx, &b, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(1..3), RangeArg::from(3)];
    let btv = bt.slice(slice);

    let c = Array::from_vec(vec![4, 4], vec![0; 16]);
    let ct = Tensor::from_array(ctx, &c, TensorMode::Mut);
    let slice: &[RangeArg] = &[RangeArg::from(2..4), RangeArg::from(0)];
    let ctv = ct.slice(slice);

    add_slice(ctx, &atv, &btv, &ctv);
    //println!("{:?}", ct.get(ctx));
    assert!(ct.get(ctx).buffer() == &[0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      15, 0, 0, 0,
                                      23, 0, 0, 0]);
}
