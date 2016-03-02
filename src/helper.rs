pub fn compute_dim_steps(shape: &[usize]) -> Vec<usize> {
    let mut dim_steps = vec![0; shape.len()];
    dim_steps[shape.len()-1] = 1;
    for i in 1..shape.len() {
        let cur_index = shape.len()-i-1;
        dim_steps[cur_index] = shape[cur_index+1]*dim_steps[cur_index+1];
    }
    dim_steps
}
