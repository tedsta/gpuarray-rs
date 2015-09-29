use std::ops::Index;

use num::Num;

// A 2D matrix
pub struct Matrix2<T: Num> {
    rows: usize,
    columns: usize,
    buffer: Vec<T>,
}

impl<T: Num> Matrix2<T> {
    pub fn new(rows: usize, columns: usize, initial: T) -> Matrix2<T> {
        Matrix2 {
            rows: rows,
            columns: columns,
            buffer: vec![initial; rows*columns],
        }
    }

    pub fn from_vec(rows: usize, columns: usize, vec: Vec<T>) -> Matrix2<T> {
        Matrix2 {
            rows: rows,
            columns: columns,
            buffer: vec,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> &T {
        &self.buffer[row*self.columns + column]
    }

    pub fn get_mut(&mut self, row: usize, column: usize) -> &mut T {
        &mut self.buffer[row*self.columns + column]
    }
    
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn buffer(&self) -> &Vec<T> {
        &self.buffer
    }
}

impl<T: Num> Index<(usize, usize)> for Matrix2<T> {
    type Output = T;

    fn index<'a>(&'a self, index: (usize, usize)) -> &'a T {
        self.get(index.0, index.1)
    }
}
