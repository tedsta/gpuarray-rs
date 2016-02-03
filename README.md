# matrix-rs

Make use of GPU-powered matrix operations from Rust! Matrix-rs uses OpenCL but hides all the details. The API is still very crude. In the future it may be generalized to multiple backends.

### Road map

- Rethink naming. ClMatrix needs to be renamed if we generalize to multiple backends.
- Make operations pluggable rather than having methods on ClMatrix.
- Generalize to multiple backends

### Example

A simple dot product between two matrices:

```Rust
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
    let mut c_cl: ClMatrix<f32> = ClMatrix::new(ctx, 5, 15, ClMatrixMode::Mut);

    a_cl.dot(ctx, &b_cl, &mut c_cl);
    
    let c = c_cl.get(ctx);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("A*B = \n{:?}", c);
}
```

### License

MIT
