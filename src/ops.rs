use std::cell::Ref;

use context::Context;
use num::Num;
use tensor::{Event, Tensor, TensorView};
use range_arg::RangeArg;

pub fn copy_to<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().copy_to::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn fill<T: Num>(ctx: &Context, a: &Tensor<T>, val: T) {
    let kernel = ctx.kernels().fill::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, &val);

    a.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(), None, ()));
}

pub fn sum<T: Num>(ctx: &Context, a: &Tensor<T>, axis: usize, b: &Tensor<T>) {
    let kernel = ctx.kernels().sum::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a.shape()[0]);
    kernel.set_arg(3, &a.shape()[1]);
    kernel.set_arg(4, &axis);

    let keep_dim = [a.shape()[1], a.shape()[0]][axis];

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, keep_dim, None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

pub fn add<T: Num>(ctx: &Context, a: &Tensor<T>, axis: i32, b: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().add::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, output);
    kernel.set_arg(3, &a.shape()[1]);
    kernel.set_arg(4, &axis);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel, (a.shape()[0], a.shape()[1]), None, event_list)
    };
    output.set_event(new_event);
}

pub fn sub<T: Num>(ctx: &Context, a: &Tensor<T>, b: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().sub::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, output);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel, a.len(), None, event_list)
    };
    output.set_event(new_event);
}

pub fn multiply<T: Num>(ctx: &Context, a: &Tensor<T>, axis: i32, b: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().multiply::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, output);
    kernel.set_arg(3, &a.shape()[1]);
    kernel.set_arg(4, &axis);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel, (a.shape()[0], a.shape()[1]), None, event_list)
    };
    output.set_event(new_event);
}

pub fn transpose<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().transpose::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);
    kernel.set_arg(2, &a.shape()[0]);
    kernel.set_arg(3, &a.shape()[1]);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, (a.shape()[0], a.shape()[1]),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn matmul<T: Num>(ctx: &Context, a: &Tensor<T>, b: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().matmul::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, output);
    kernel.set_arg(3, &a.shape()[1]);
    kernel.set_arg(4, &b.shape()[1]);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel,
                                       (a.shape()[0], b.shape()[1]),
                                       None, event_list)
    };
    output.set_event(new_event);
}

pub fn max<T: Num>(ctx: &Context, a: &Tensor<T>, threshold: T, output: &Tensor<T>) {
    let kernel = ctx.kernels().max::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);
    kernel.set_arg(2, &threshold);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn dmax<T: Num>(ctx: &Context, a: &Tensor<T>, threshold: T, output: &Tensor<T>) {
    let kernel = ctx.kernels().dmax::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);
    kernel.set_arg(2, &threshold);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn min<T: Num>(ctx: &Context, a: &Tensor<T>, threshold: T, output: &Tensor<T>) {
    let kernel = ctx.kernels().min::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);
    kernel.set_arg(2, &threshold);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn dmin<T: Num>(ctx: &Context, a: &Tensor<T>, threshold: T, output: &Tensor<T>) {
    let kernel = ctx.kernels().dmin::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);
    kernel.set_arg(2, &threshold);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn mse<T: Num>(ctx: &Context, a: &Tensor<T>, train: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().mse::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, train);
    kernel.set_arg(2, output);
    kernel.set_arg(3, &a.shape()[0]);
    kernel.set_arg(4, &a.shape()[1]);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), train.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel,
                                       a.shape()[1],
                                       None, event_list)
    };
    output.set_event(new_event);
}

pub fn dmse<T: Num>(ctx: &Context, a: &Tensor<T>, train: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().dmse::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, train);
    kernel.set_arg(2, output);
    kernel.set_arg(3, &a.shape()[0]);
    kernel.set_arg(4, &a.shape()[1]);

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), train.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel,
                                       (a.shape()[0], a.shape()[1]),
                                       None, event_list)
    };
    output.set_event(new_event);
}

pub fn tanh<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().tanh::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn dtanh<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().dtanh::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn sigmoid<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().sigmoid::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn dsigmoid<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().dsigmoid::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn log<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().log::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

pub fn exp<T: Num>(ctx: &Context, a: &Tensor<T>, output: &Tensor<T>) {
    let kernel = ctx.kernels().exp::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, output);

    output.set_event(ctx.queue.enqueue_async_kernel(&kernel, a.len(),
                                                    None, a.get_event().as_ref().map(|x| &**x)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn dim_steps_as_ulong4(dim_steps: &[usize]) -> [u64; 4] {
    let mut array = [0u64; 4];
    let array_len = array.len();

    for i in 0..dim_steps.len() {
        array[array_len-1-i] = dim_steps[dim_steps.len()-1-i] as u64;
    }
    for i in dim_steps.len()..array_len {
        array[array_len-1-i] = dim_steps[0] as u64;
    }

    array
}

fn tensor_view_offsets_as_ulong4<T: Num, R: AsRef<[RangeArg]>>(t: &TensorView<T, R>) -> [u64; 4] {
    let mut array = [0u64; 4];
    let array_len = array.len();

    for i in 0..t.shape.len() {
        array[array_len-1-i] = t.view_offset(t.shape.len()-1-i) as u64;
    }
    for i in t.shape.len()..array_len {
        array[array_len-1-i] = 0;
    }

    array
}

pub fn fill_slice<T: Num, AR: AsRef<[RangeArg]>>(ctx: &Context, a: &TensorView<T, AR>, val: T) {
    let kernel = ctx.kernels().fill_slice::<T>();

    let a_dim_steps = dim_steps_as_ulong4(a.dim_steps);
    let a_offsets = tensor_view_offsets_as_ulong4(a);

    kernel.set_arg(0, a);
    kernel.set_arg(1, &val);
    kernel.set_arg(2, &a_dim_steps);
    kernel.set_arg(3, &a_offsets);
    
    let mut work_dim = [1; 3];
    for i in 0..a.shape.len() {
        work_dim[2-i] = a.view_shape(a.shape.len()-1-i);
    }

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, work_dim, None, a.get_event().as_ref().map(|x| &**x))
    };
    a.set_event(new_event);
}

pub fn copy_to_slice<T: Num, AR, BR>(ctx: &Context,
                                     a: &TensorView<T, AR>,
                                     b: &TensorView<T, BR>)
                                     where AR: AsRef<[RangeArg]>,
                                           BR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().copy_to_slice::<T>();

    let a_dim_steps = dim_steps_as_ulong4(a.dim_steps);
    let b_dim_steps = dim_steps_as_ulong4(b.dim_steps);

    let a_offsets = tensor_view_offsets_as_ulong4(a);
    let b_offsets = tensor_view_offsets_as_ulong4(b);

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a_dim_steps);
    kernel.set_arg(3, &a_offsets);
    kernel.set_arg(4, &b_dim_steps);
    kernel.set_arg(5, &b_offsets);
    
    let mut work_dim = [1; 3];
    for i in 0..a.shape.len() {
        work_dim[2-i] = a.view_shape(a.shape.len()-1-i);
    }

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, work_dim, None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

pub fn add_slice<T: Num, AR, BR, CR>(ctx: &Context,
                                     a: &TensorView<T, AR>,
                                     b: &TensorView<T, BR>,
                                     out: &TensorView<T, CR>)
                                     where AR: AsRef<[RangeArg]>,
                                           BR: AsRef<[RangeArg]>,
                                           CR: AsRef<[RangeArg]>
{
    let kernel = ctx.kernels().add_slice::<T>();

    let a_dim_steps = dim_steps_as_ulong4(a.dim_steps);
    let b_dim_steps = dim_steps_as_ulong4(b.dim_steps);
    let out_dim_steps = dim_steps_as_ulong4(out.dim_steps);

    let a_offsets = tensor_view_offsets_as_ulong4(a);
    let b_offsets = tensor_view_offsets_as_ulong4(b);
    let out_offsets = tensor_view_offsets_as_ulong4(out);

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, out);
    kernel.set_arg(3, &a_dim_steps);
    kernel.set_arg(4, &a_offsets);
    kernel.set_arg(5, &b_dim_steps);
    kernel.set_arg(6, &b_offsets);
    kernel.set_arg(7, &out_dim_steps);
    kernel.set_arg(8, &out_offsets);
    
    let mut work_dim = [1; 3];
    for i in 0..a.shape.len() {
        work_dim[2-i] = a.view_shape(a.shape.len()-1-i);
    }

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel, work_dim, None, event_list)
    };
    out.set_event(new_event);
}

pub fn multiply_slice<T: Num, AR, BR, CR>(ctx: &Context,
                                      a: &TensorView<T, AR>,
                                      b: &TensorView<T, BR>,
                                      out: &TensorView<T, CR>)
                                      where AR: AsRef<[RangeArg]>,
                                            BR: AsRef<[RangeArg]>,
                                            CR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().multiply_slice::<T>();

    let a_dim_steps = dim_steps_as_ulong4(a.dim_steps);
    let b_dim_steps = dim_steps_as_ulong4(b.dim_steps);
    let out_dim_steps = dim_steps_as_ulong4(out.dim_steps);

    let a_offsets = tensor_view_offsets_as_ulong4(a);
    let b_offsets = tensor_view_offsets_as_ulong4(b);
    let out_offsets = tensor_view_offsets_as_ulong4(out);

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, out);
    kernel.set_arg(3, &a_dim_steps);
    kernel.set_arg(4, &a_offsets);
    kernel.set_arg(5, &b_dim_steps);
    kernel.set_arg(6, &b_offsets);
    kernel.set_arg(7, &out_dim_steps);
    kernel.set_arg(8, &out_offsets);
    
    let mut work_dim = [1; 3];
    for i in 0..a.shape.len() {
        work_dim[2-i] = a.view_shape(a.shape.len()-1-i);
    }

    let new_event = {
        let event_list: &[Option<Ref<Event>>] = &[a.get_event(), b.get_event()];
        ctx.queue.enqueue_async_kernel(&kernel, work_dim, None, event_list)
    };
    out.set_event(new_event);
}

pub fn sigmoid_slice<T: Num, AR, BR>(ctx: &Context,
                                     a: &TensorView<T, AR>,
                                     b: &TensorView<T, BR>)
                                     where AR: AsRef<[RangeArg]>,
                                           BR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().sigmoid_slice::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a.view_offset(0));
    kernel.set_arg(3, &a.view_offset(1));
    kernel.set_arg(4, &b.view_offset(0));
    kernel.set_arg(5, &b.view_offset(1));
    kernel.set_arg(6, &a.shape[1]);
    kernel.set_arg(7, &b.shape[1]);

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, (a.view_shape(0), a.view_shape(1)), None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

pub fn dsigmoid_slice<T: Num, AR, BR>(ctx: &Context,
                                      a: &TensorView<T, AR>,
                                      b: &TensorView<T, BR>)
                                      where AR: AsRef<[RangeArg]>,
                                           BR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().dsigmoid_slice::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a.view_offset(0));
    kernel.set_arg(3, &a.view_offset(1));
    kernel.set_arg(4, &b.view_offset(0));
    kernel.set_arg(5, &b.view_offset(1));
    kernel.set_arg(6, &a.shape[1]);
    kernel.set_arg(7, &b.shape[1]);

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, (a.view_shape(0), a.view_shape(1)), None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

pub fn tanh_slice<T: Num, AR, BR>(ctx: &Context,
                                  a: &TensorView<T, AR>,
                                  b: &TensorView<T, BR>)
                                  where AR: AsRef<[RangeArg]>,
                                        BR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().tanh_slice::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a.view_offset(0));
    kernel.set_arg(3, &a.view_offset(1));
    kernel.set_arg(4, &b.view_offset(0));
    kernel.set_arg(5, &b.view_offset(1));
    kernel.set_arg(6, &a.shape[1]);
    kernel.set_arg(7, &b.shape[1]);

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, (a.view_shape(0), a.view_shape(1)), None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

pub fn dtanh_slice<T: Num, AR, BR>(ctx: &Context,
                                   a: &TensorView<T, AR>,
                                   b: &TensorView<T, BR>)
                                   where AR: AsRef<[RangeArg]>,
                                         BR: AsRef<[RangeArg]>,
{
    let kernel = ctx.kernels().dtanh_slice::<T>();

    kernel.set_arg(0, a);
    kernel.set_arg(1, b);
    kernel.set_arg(2, &a.view_offset(0));
    kernel.set_arg(3, &a.view_offset(1));
    kernel.set_arg(4, &b.view_offset(0));
    kernel.set_arg(5, &b.view_offset(1));
    kernel.set_arg(6, &a.shape[1]);
    kernel.set_arg(7, &b.shape[1]);

    let new_event = {
        ctx.queue.enqueue_async_kernel(&kernel, (a.view_shape(0), a.view_shape(1)), None, a.get_event().as_ref().map(|x| &**x))
    };
    b.set_event(new_event);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
use array::Array;
#[cfg(test)]
use tensor::TensorMode;

#[test]
fn tensor_fill() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], vec![0i32; 15]);

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);

    fill(ctx, &a_cl, 42);
    let result = a_cl.get(ctx);

    assert!(result.buffer() == &[42; 15]);
}

#[test]
fn tensor_transpose() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], (0..15).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::new(ctx, vec![3, 5], TensorMode::Out);

    transpose(ctx, &a_cl, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[0, 3, 6, 9, 12,
                            1, 4, 7, 10, 13,
                            2, 5, 8, 11, 14]);
}

#[test]
fn tensor_sum_axis0() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![1, 3], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::Out);

    sum(ctx, &a_cl, 0, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[30, 35, 40]);
}

#[test]
fn tensor_sum_axis1() {
    let ref ctx = Context::new();

    let a: Array<i32> = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![5, 1], (0..5).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::Out);

    sum(ctx, &a_cl, 1, &b_cl);
    let b = b_cl.get(ctx);

    assert!(b.buffer() == &[3, 12, 21, 30, 39]);
}

#[test]
fn tensor_add() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 10000], (0..5*10000).collect());
    let b = Array::from_vec(vec![5, 10000], (0..5*10000).map(|x| x*2).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_cl: Tensor<i32> = Tensor::new(ctx, vec![5, 10000], TensorMode::Out);

    add(ctx, &a_cl, -1, &b_cl, &c_cl);
    
    let c = c_cl.get(ctx);

    for i in 0..5 {
        for j in 0..10000 {
            assert!(c[&[i, j]] == a[&[i, j]] + b[&[i, j]]);
        }
    }
}

#[test]
fn tensor_add_reuse() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![1, 10000], (0..10000).collect());
    let b = Array::from_vec(vec![1, 10000], (0..10000).map(|x| x*2).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    add(ctx, &a_cl, -1, &b_cl, &b_cl); // b = a+b
}

#[test]
fn tensor_add_axis() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 3], (0..15).collect());
    let b = Array::from_vec(vec![1, 3], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    add(ctx, &a_cl, 0, &b_cl, &a_cl); // a = a+b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0, 2, 4,
                            3, 5, 7,
                            6, 8, 10,
                            9, 11, 13,
                            12, 14, 16]);
}

#[test]
fn tensor_multiply_axis1() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![3, 5], (0..15).collect());
    let b = Array::from_vec(vec![3, 1], (0..3).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);

    multiply(ctx, &a_cl, 1, &b_cl, &a_cl); // a = a*b

    let a = a_cl.get(ctx);

    assert!(a.buffer() == &[0,  0,  0,  0,  0,
                            5,  6,  7,  8,  9,
                            20, 22, 24, 26, 28]);
}

#[test]
fn tensor_matmul() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![3, 5], (0i32..15).collect());
    let b = Array::from_vec(vec![5, 2], (0..10).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let b_cl = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_cl = Tensor::new(ctx, vec![3, 2], TensorMode::In);

    matmul(ctx, &a_cl, &b_cl, &c_cl); // c = a*b

    let c = c_cl.get(ctx);

    println!("{:?}", c);

    assert!(c.buffer() == &[60, 70,
                            160, 195,
                            260, 320]);
}

#[test]
fn tensor_tanh() {
    let ref ctx = Context::new();

    let a: Array<f32> = Array::from_vec(vec![5, 3], (0..15).map(|x| x as f32).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::new(ctx, vec![5, 3], TensorMode::Out);

    tanh(ctx, &a_cl, &b_cl);
    let b = b_cl.get(ctx);

    println!("{:?}", b);
    assert!(b.buffer() == &[0.0, 0.7615941, 0.9640276,
                            0.9950548, 0.9993293, 0.99990916,
                            0.99998766, 0.99999833, 0.99999976,
                            1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0]);

}

#[test]
fn tensor_dtanh() {
    let ref ctx = Context::new();

    let a: Array<f32> = Array::from_vec(vec![5, 3], (0..15).map(|x| x as f32).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::new(ctx, vec![5, 3], TensorMode::Out);

    dtanh(ctx, &a_cl, &b_cl);
    let b = b_cl.get(ctx);

    println!("{:?}", b);
    assert!(b.buffer() == &[1.0, 0.4199744, 0.070650816,
                            0.009865999, 0.0013408661, 0.00018167496,
                            0.000024676323, 0.00000333786, 0.00000047683716,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]);

}

#[test]
fn tensor_sigmoid() {
    let ref ctx = Context::new();

    let a: Array<f32> = Array::from_vec(vec![5, 3], (0..15).map(|x| x as f32).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::new(ctx, vec![5, 3], TensorMode::Out);

    sigmoid(ctx, &a_cl, &b_cl);
    let b = b_cl.get(ctx);

    println!("{:?}", b);
    assert!(b.buffer() == &[0.5, 0.7310586, 0.880797,
                            0.9525742, 0.98201376, 0.9933072,
                            0.9975274, 0.999089, 0.99966466,
                            0.9998766, 0.9999546, 0.9999833,
                            0.9999938, 0.99999774, 0.99999917]);

}

#[test]
fn tensor_dsigmoid() {
    let ref ctx = Context::new();

    let a: Array<f32> = Array::from_vec(vec![5, 3], (0..15).map(|x| x as f32).collect());

    let a_cl = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_cl = Tensor::new(ctx, vec![5, 3], TensorMode::Out);

    dsigmoid(ctx, &a_cl, &b_cl);
    let b = b_cl.get(ctx);

    println!("{:?}", b);
    assert!(b.buffer() == &[0.25, 0.19661193, 0.10499363,
                            0.0451766, 0.017662734, 0.006648033,
                            0.0024664658, 0.00091016747, 0.00033522327,
                            0.0001233664, 0.000045416677, 0.000016689022,
                            0.000006198845, 0.0000022649713, 0.00000083446434]);
}

#[test]
fn test_add_slice() {
    use array::Array;
    use context::Context;
    use tensor::{Tensor, TensorMode};

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![2, 4, 3], vec![2, 3, 4,
                                                6, 7, 8,
                                                10, 11, 12,
                                                14, 15, 16,
                                                
                                                17, 18, 19,
                                                20, 21, 22,
                                                23, 24, 25,
                                                26, 27, 28]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let atv = at.slice(s![0, 1..3, 1]);

    let b = Array::from_vec(vec![4, 4], vec![1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16]);
    let bt = Tensor::from_array(ctx, &b, TensorMode::Mut);
    let btv = bt.slice(s![1..3, 3]);

    let c = Array::from_vec(vec![4, 4], vec![0; 16]);
    let ct = Tensor::from_array(ctx, &c, TensorMode::Mut);
    let ctv = ct.slice(s![2..4, 0]);

    add_slice(ctx, &btv, &atv, &ctv);
    println!("{:?}", ct.get(ctx));
    assert!(ct.get(ctx).buffer() == &[0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      15, 0, 0, 0,
                                      23, 0, 0, 0]);
}

#[test]
fn test_multiply_slice() {
    use array::Array;
    use context::Context;
    use tensor::{Tensor, TensorMode};

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![2, 4, 3], vec![2, 3, 4,
                                                6, 7, 8,
                                                10, 11, 12,
                                                14, 15, 16,
                                                
                                                17, 18, 19,
                                                20, 21, 22,
                                                23, 24, 25,
                                                26, 27, 28]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let atv = at.slice(s![0, 1..3, 1]);

    let b = Array::from_vec(vec![4, 4], vec![1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16]);
    let bt = Tensor::from_array(ctx, &b, TensorMode::Mut);
    let btv = bt.slice(s![1..3, 3]);

    let c = Array::from_vec(vec![4, 4], vec![0; 16]);
    let ct = Tensor::from_array(ctx, &c, TensorMode::Mut);
    let ctv = ct.slice(s![2..4, 0]);

    multiply_slice(ctx, &btv, &atv, &ctv);
    println!("{:?}", ct.get(ctx));
    assert!(ct.get(ctx).buffer() == &[0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      56, 0, 0, 0,
                                      132, 0, 0, 0]);
}

#[test]
fn test_fill_slice() {
    use array::Array;
    use context::Context;
    use tensor::{Tensor, TensorMode};

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![4, 3], vec![0i32; 12]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let atv = at.slice(s![1..3, 1]);

    fill_slice(ctx, &atv, 42);
    assert!(at.get(ctx).buffer() == &[0, 0, 0,
                                      0, 42, 0,
                                      0, 42, 0,
                                      0, 0, 0]);
}

#[test]
fn test_copy_to_slice() {
    use array::Array;
    use context::Context;
    use tensor::{Tensor, TensorMode};

    let ref ctx = Context::new();

    let a = Array::from_vec(vec![4, 3], vec![2i32, 3, 4,
                                             6, 7, 8,
                                             10, 11, 12,
                                             14, 15, 16]);
    let at = Tensor::from_array(ctx, &a, TensorMode::Mut);
    let atv = at.slice(s![1..3, 1]);

    let b = Array::from_vec(vec![4, 4], vec![0; 16]);
    let bt = Tensor::from_array(ctx, &b, TensorMode::Mut);
    let btv = bt.slice(s![2..4, 3]);

    copy_to_slice(ctx, &atv, &btv);
    assert!(bt.get(ctx).buffer() == &[0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0, 7,
                                      0, 0, 0, 11]);
}
