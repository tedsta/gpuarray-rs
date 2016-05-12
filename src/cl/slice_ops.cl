ulong index2(ulong cols, ulong row, ulong col) {
    return row*cols + col;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_add_slice_i32(__global int* a, __global int* b, __global int* c,
                                  ulong a_off0, ulong a_off1,
                                  ulong b_off0, ulong b_off1,
                                  ulong c_off0, ulong c_off1,
                                  ulong a_cols, ulong b_cols, ulong c_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    ulong ci = index2(c_cols, i+c_off0, j+c_off1);
    c[ci] = a[ai] + b[bi];
}

__kernel void array_add_slice_f32(__global float* a, __global float* b, __global float* c,
                                  ulong a_off0, ulong a_off1,
                                  ulong b_off0, ulong b_off1,
                                  ulong c_off0, ulong c_off1,
                                  ulong a_cols, ulong b_cols, ulong c_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    ulong ci = index2(c_cols, i+c_off0, j+c_off1);
    c[ci] = a[ai] + b[bi];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_copy_to_slice_i32(__global int* a, __global int* b,
                                      ulong a_off0, ulong a_off1,
                                      ulong b_off0, ulong b_off1,
                                      ulong a_cols, ulong b_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    b[bi] = a[ai];
}

__kernel void array_copy_to_slice_f32(__global float* a, __global float* b,
                                      ulong a_off0, ulong a_off1,
                                      ulong b_off0, ulong b_off1,
                                      ulong a_cols, ulong b_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    b[bi] = a[ai];
}
