use std::fmt;
use std::ops::Index;

use num::Num;

// A 2D array
pub struct Array<T: Num> {
    shape: Vec<usize>,
    dim_steps: Vec<usize>, // the ‘volume’ of 1 unit in each dimension.
    buffer: Vec<T>,
}

impl<T: Num> Array<T> {
    pub fn new(shape: Vec<usize>, initial: T) -> Array<T> {
        let buf_size = shape.iter().fold(1, |a, b| a*b);
        let dim_steps = compute_dim_steps(&shape);
        Array {
            shape: shape,
            dim_steps: dim_steps,
            buffer: vec![initial; buf_size],
        }
    }

    pub fn from_vec(shape: Vec<usize>, vec: Vec<T>) -> Array<T> {
        let dim_steps = compute_dim_steps(&shape);
        Array {
            shape: shape,
            dim_steps: dim_steps,
            buffer: vec,
        }
    }

    pub fn get<'a>(&'a self, coords: &[usize]) -> &'a T {
        let index: usize = coords.iter()
                                 .zip(self.dim_steps.iter())
                                 .map(|(c, s)| (*c)*(*s))
                                 .sum();
        &self.buffer[index]
    }

    pub fn get_mut<'a>(&'a mut self, coords: &[usize]) -> &'a mut T {
        let index: usize = coords.iter()
                                 .zip(self.dim_steps.iter())
                                 .map(|(c, s)| (*c)*(*s))
                                 .sum();
        &mut self.buffer[index]
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn buffer(&self) -> &Vec<T> {
        &self.buffer
    }
}

impl<'a, T: Num> Index<&'a [usize]> for Array<T> {
    type Output = T;

    fn index<'b>(&'b self, index: &'a [usize]) -> &'b T {
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

fn compute_dim_steps(shape: &[usize]) -> Vec<usize> {
    let mut dim_steps = vec![0; shape.len()];
    dim_steps[shape.len()-1] = 1;
    for i in 1..shape.len() {
        let cur_index = shape.len()-i-1;
        dim_steps[cur_index] = shape[cur_index]*shape[cur_index+1];
    }
    dim_steps
}
