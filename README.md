# gpuarray-rs

Make use of GPU-powered array operations from Rust! Gpuarray-rs uses OpenCL but hides all the details. Still mostly a proof of concept.

### Example

Matrix multiplication

```Rust
extern crate gpuarray as ga;

use ga::Context;
use ga::tensor::{Tensor, TensorMode};
use ga::array::Array;

fn main() {
    let ref ctx = Context::new();

    let a = Array::from_vec(vec![5, 10], (0..5*10).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![10, 15], (0..10*15).map(|x| (x as f32)*2.0).collect());

    let a_gpu = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_gpu = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_gpu: Tensor<f32> = Tensor::new(ctx, vec![5, 15], TensorMode::Mut);

    ga::matmul(ctx, &a_gpu, &b_gpu, &c_gpu);
    
    let c = c_gpu.get(ctx);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("A*B = \n{:?}", c);
}
```

### License

MIT
