ulong index2(ulong cols, ulong row, ulong col) {
    return row*cols + col;
}

ulong dot_ulong4(ulong4 a, ulong4 b) {
    ulong4 prod = a*b;
    ulong2 half_sum = prod.xy + prod.zw;
    return half_sum[0] + half_sum[1];
}

ulong index4(ulong4 dim_steps, ulong4 coords) {
    return dot_ulong4(dim_steps, coords);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_add_slice_i32(__global int* a, __global int* b, __global int* c,
                                  ulong4 a_dim_steps, ulong4 a_off,
                                  ulong4 b_dim_steps, ulong4 b_off,
                                  ulong4 c_dim_steps, ulong4 c_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    c_off[1] += i;
    c_off[2] += j;
    c_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);
    ulong ci = index4(c_dim_steps, c_off);

    c[ci] = a[ai] + b[bi];
}

__kernel void array_add_slice_f32(__global float* a, __global float* b, __global float* c,
                                  ulong4 a_dim_steps, ulong4 a_off,
                                  ulong4 b_dim_steps, ulong4 b_off,
                                  ulong4 c_dim_steps, ulong4 c_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    c_off[1] += i;
    c_off[2] += j;
    c_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);
    ulong ci = index4(c_dim_steps, c_off);

    c[ci] = a[ai] + b[bi];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_multiply_slice_i32(__global int* a, __global int* b, __global int* c,
                                       ulong4 a_dim_steps, ulong4 a_off,
                                       ulong4 b_dim_steps, ulong4 b_off,
                                       ulong4 c_dim_steps, ulong4 c_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    c_off[1] += i;
    c_off[2] += j;
    c_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);
    ulong ci = index4(c_dim_steps, c_off);

    c[ci] = a[ai] * b[bi];
}

__kernel void array_multiply_slice_f32(__global float* a, __global float* b, __global float* c,
                                       ulong4 a_dim_steps, ulong4 a_off,
                                       ulong4 b_dim_steps, ulong4 b_off,
                                       ulong4 c_dim_steps, ulong4 c_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    c_off[1] += i;
    c_off[2] += j;
    c_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);
    ulong ci = index4(c_dim_steps, c_off);

    c[ci] = a[ai] * b[bi];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_copy_to_slice_i32(__global int* a, __global int* b,
                                      ulong4 a_dim_steps, ulong4 a_off,
                                      ulong4 b_dim_steps, ulong4 b_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);

    b[bi] = a[ai];
}

__kernel void array_copy_to_slice_f32(__global float* a, __global float* b,
                                      ulong4 a_dim_steps, ulong4 a_off,
                                      ulong4 b_dim_steps, ulong4 b_off) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong k = get_global_id(2);

    a_off[1] += i;
    a_off[2] += j;
    a_off[3] += k;

    b_off[1] += i;
    b_off[2] += j;
    b_off[3] += k;

    ulong ai = index4(a_dim_steps, a_off);
    ulong bi = index4(b_dim_steps, b_off);

    b[bi] = a[ai];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_sigmoid_slice_f32(__global float* a, __global float* b,
                                      ulong a_off0, ulong a_off1,
                                      ulong b_off0, ulong b_off1,
                                      ulong a_cols, ulong b_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    b[bi] = sigmoid(a[ai]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_tanh_slice_f32(__global float* a, __global float* b,
                                   ulong a_off0, ulong a_off1,
                                   ulong b_off0, ulong b_off1,
                                   ulong a_cols, ulong b_cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    ulong ai = index2(a_cols, i+a_off0, j+a_off1);
    ulong bi = index2(b_cols, i+b_off0, j+b_off1);
    b[bi] = tanh(a[ai]);
}
