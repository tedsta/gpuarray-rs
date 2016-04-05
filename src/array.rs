use std::fmt;
use std::ops::Index;

use num::Num;
use helper;

// A n-dimensional array
pub struct Array<T: Num> {
    shape: Vec<usize>,
    dim_steps: Vec<usize>, // the ‘volume’ of 1 unit in each dimension.
    buffer: Vec<T>,
}

impl<T: Num> Array<T> {
    pub fn new(shape: Vec<usize>, initial: T) -> Array<T> {
        let buf_size = shape.iter().fold(1, |a, b| a*b);
        let dim_steps = helper::compute_dim_steps(&shape);
        Array {
            shape: shape,
            dim_steps: dim_steps,
            buffer: vec![initial; buf_size],
        }
    }

    pub fn from_vec(shape: Vec<usize>, vec: Vec<T>) -> Array<T> {
        let dim_steps = helper::compute_dim_steps(&shape);
        Array {
            shape: shape,
            dim_steps: dim_steps,
            buffer: vec,
        }
    }

    pub fn get<'a, 'b, I: IntoIterator<Item=&'b usize>>(&'a self, coords: I) -> &'a T {
        let index: usize = coords.into_iter().zip(self.dim_steps.iter())
                                 .map(|(c, s)| (*c)*(*s))
                                 .sum();
        &self.buffer[index]
    }

    pub fn get_mut<'a, 'b, I: IntoIterator<Item=&'b usize>>(&'a mut self, coords: I) -> &'a mut T {
        let index: usize = coords.into_iter().zip(self.dim_steps.iter())
                                 .map(|(c, s)| (*c)*(*s))
                                 .sum();
        &mut self.buffer[index]
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim_steps(&self) -> &[usize] {
        &self.shape
    }

    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [T] {
        &mut self.buffer
    }
}

impl<'a, 'b, T: Num, I: IntoIterator<Item=&'b usize>> Index<I> for Array<T> {
    type Output = T;

    fn index<'r>(&'r self, index: I) -> &'r T {
        self.get(index)
    }
}

impl<T: Num> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "[\n"));
        for row in 0..self.shape[0] {
            try!(write!(f, "[{}", self.get(&[row, 0])));
            for col in 1..self.shape[1] {
                try!(write!(f, "\t{}", self.get(&[row, col])));
            }
            try!(write!(f, "]\n"));
        }
        try!(write!(f, "]\n"));
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_array_indexing() {
    let a = Array::from_vec(vec![2, 3, 4], vec![1, 2, 3, 4,
                                                5, 6, 7, 8,
                                                9, 10, 11, 12,

                                                13, 14, 15, 16,
                                                17, 18, 19, 20,
                                                21, 22, 23, 24]);
    assert!(a[&[0, 0, 0]] == 1);
    assert!(a[&[0, 0, 1]] == 2);
    assert!(a[&[0, 0, 2]] == 3);
    assert!(a[&[0, 0, 3]] == 4);
    assert!(a[&[0, 1, 0]] == 5);
    assert!(a[&[0, 1, 1]] == 6);
    assert!(a[&[0, 1, 2]] == 7);
    assert!(a[&[0, 1, 3]] == 8);
    assert!(a[&[0, 2, 0]] == 9);
    assert!(a[&[0, 2, 1]] == 10);
    assert!(a[&[0, 2, 2]] == 11);
    assert!(a[&[0, 2, 3]] == 12);
    assert!(a[&[1, 0, 0]] == 13);
    assert!(a[&[1, 0, 1]] == 14);
    assert!(a[&[1, 0, 2]] == 15);
    assert!(a[&[1, 0, 3]] == 16);
    assert!(a[&[1, 1, 0]] == 17);
    assert!(a[&[1, 1, 1]] == 18);
    assert!(a[&[1, 1, 2]] == 19);
    assert!(a[&[1, 1, 3]] == 20);
    assert!(a[&[1, 2, 0]] == 21);
    assert!(a[&[1, 2, 1]] == 22);
    assert!(a[&[1, 2, 2]] == 23);
    assert!(a[&[1, 2, 3]] == 24);
}
