__kernel void array_fill_f32(__global float* a, float val) {
    uintptr_t i = get_global_id(0);
    a[i] = val;
}

__kernel void array_fill_i32(__global int* a, int val) {
    uintptr_t i = get_global_id(0);
    a[i] = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_copy_to_f32(__global const float *f,
                                __global float *t) {
    uintptr_t i = get_global_id(0);
    t[i] = f[i];
}

__kernel void array_copy_to_i32(__global const int *f,
                                __global int *t) {
    uintptr_t i = get_global_id(0);
    t[i] = f[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sum

__kernel void array_sum_f32(__global float *a,
                            __global float *b,
                            ulong rows,
                            ulong cols,
                            ulong axis) {
    ulong i = get_global_id(0);

    b[i] = 0.0; // Initialize to zero before we start summing things up
    
    if (axis == 0) {
        for (ulong m = 0; m < rows; m++) {
            b[i] += a[m*cols + i];
        }
    } else if (axis == 1) {
        for (ulong m = 0; m < cols; m++) {
            b[i] += a[i*cols + m];
        }
    }
}

__kernel void array_sum_i32(__global int *a,
                            __global int *b,
                            ulong rows,
                            ulong cols,
                            ulong axis) {
    ulong i = get_global_id(0);

    b[i] = 0.0; // Initialize to zero before we start summing things up
    
    if (axis == 0) {
        for (ulong m = 0; m < rows; m++) {
            b[i] += a[m*cols + i];
        }
    } else if (axis == 1) {
        for (ulong m = 0; m < cols; m++) {
            b[i] += a[i*cols + m];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_add_f32(__global const float *a,
                             __global const float *b,
                             __global float *c,
                             const ulong cols,
                             const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] + b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] + b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] + b[i];
    }
}

__kernel void array_add_i32(__global const int *a,
                             __global const int *b,
                             __global int *c,
                             const ulong cols,
                             const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] + b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] + b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] + b[i];
    }
}

__kernel void array_add_u64(__global const ulong *a,
                            __global const ulong *b,
                            __global ulong *c,
                            const ulong cols,
                            const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] + b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] + b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] + b[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_sub_f32(__global const float *a,
                            __global const float *b,
                            __global float *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_i8(__global const char *a,
                           __global const char *b,
                           __global char *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_i16(__global const short *a,
                            __global const short *b,
                            __global short *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_i32(__global const int *a,
                            __global const int *b,
                            __global int *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_i64(__global const long *a,
                            __global const long *b,
                            __global long *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_u8(__global const uchar *a,
                           __global const uchar *b,
                           __global uchar *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_u16(__global const ushort *a,
                             __global const ushort *b,
                             __global ushort *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_u32(__global const uint *a,
                            __global const uint *b,
                            __global uint *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void array_sub_u64(__global const ulong *a,
                            __global const ulong *b,
                            __global ulong *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] - b[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*__kernel void array_multiply_f32(__global const float *a,
                                  __global const float *b,
                                  __global float *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}*/

__kernel void array_multiply_f32(__global const float *a,
                                  __global const float *b,
                                  __global float *c,
                                  const ulong cols,
                                  const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] * b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] * b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] * b[i];
    }
}

__kernel void array_multiply_i8(__global const char *a,
                                 __global const char *b,
                                 __global char *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_i16(__global const short *a,
                                  __global const short *b,
                                  __global short *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_i32(__global const int *a,
                                  __global const int *b,
                                  __global int *c,
                                  const ulong cols,
                                  const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] * b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] * b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] * b[i];
    }
}

__kernel void array_multiply_i64(__global const long *a,
                                  __global const long *b,
                                  __global long *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_u8(__global const uchar *a,
                                 __global const uchar *b,
                                 __global uchar *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_u16(__global const ushort *a,
                                  __global const ushort *b,
                                  __global ushort *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_u32(__global const uint *a,
                                  __global const uint *b,
                                  __global uint *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void array_multiply_u64(__global const ulong *a,
                                  __global const ulong *b,
                                  __global ulong *c) {
    uintptr_t i = get_global_id(0);
    c[i] = a[i] * b[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_divide_f32(__global const float *a,
                               __global const float *b,
                               __global float *c,
                               const ulong cols,
                               const int axis) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    if (axis == -1) {
        c[i*cols + j] = a[i*cols + j] / b[i*cols + j];
    } else if (axis == 0) {
        c[i*cols + j] = a[i*cols + j] / b[j];
    } else if (axis == 1) {
        c[i*cols + j] = a[i*cols + j] / b[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_transpose_f32(__global const float *a,
                                   __global float *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_i8(__global const char *a,
                                  __global char *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_i16(__global const short *a,
                                   __global short *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_i32(__global const int *a,
                                   __global int *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_i64(__global const long *a,
                                   __global long *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_u8(__global const uchar *a,
                                  __global uchar *b,
                                  const ulong rows,
                                  const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_u16(__global const ushort *a,
                                   __global ushort *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_u32(__global const uint *a,
                                   __global uint *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

__kernel void array_transpose_u64(__global const ulong *a,
                                   __global ulong *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void array_matmul_f32(__global const float *a,
                               __global const float *b,
                               __global float *c,
                               const ulong wa,
                               const ulong wb) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);

    float accum = 0.0;
    for (ulong k = 0; k < wa; k++) {
        accum += a[i*wa + k] * b[k*wb + j];
    }
    c[i*wb + j] = accum;
}

__kernel void array_matmul_i32(__global const int *a,
                               __global const int *b,
                               __global int *c,
                               const ulong wa,
                               const ulong wb) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);

    int accum = 0;
    for (ulong k = 0; k < wa; k++) {
        accum += a[i*wa + k] * b[k*wb + j];
    }
    c[i*wb + j] = accum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Max

__kernel void array_max_f32(__global float *a,
                             __global float *b,
                             float const threshold) {
    uintptr_t i = get_global_id(0);
    b[i] = max(threshold, a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// derivative of max with respect to a

__kernel void array_dmax_f32(__global float *a,
                              __global float *b,
                              const float threshold) {
    uintptr_t i = get_global_id(0);
    if (a[i] > threshold) {
        b[i] = 1;
    } else {
        b[i] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Min

__kernel void array_min_f32(__global float *a,
                             __global float *b,
                             const float threshold) {
    uintptr_t i = get_global_id(0);
    b[i] = min(threshold, a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// derivative of min with respect to a

__kernel void array_dmin_f32(__global float *a,
                              __global float *b,
                              const float threshold) {
    uintptr_t i = get_global_id(0);
    if (a[i] < threshold) {
        b[i] = 1;
    } else {
        b[i] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mean Squared Error (MSE)

__kernel void array_mse_f32(__global float *h,
                             __global float *y,
                             __global float *out,
                             const ulong rows,
                             const ulong cols) {
    ulong i = get_global_id(0);

    out[i] = 0.0; // Initialize to zero before we start summing things up
    
    // Sum up squared errors
    for (ulong m = 0; m < rows; m++) {
        float error = h[m*cols + i] - y[m*cols + i];
        out[i] += error*error;
    }

    // Divide by batch size to calculate the mean
    out[i] /= (float)rows;
    out[i] /= 2.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mean Squared Error derivative

__kernel void array_dmse_f32(__global float *h,
                              __global float *y,
                              __global float *out,
                              const ulong rows,
                              const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);

    uintptr_t index = i*cols + j;
    out[index] = h[index] - y[index];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tanh

__kernel void array_tanh_f32(__global float *a,
                             __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = tanh(a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// derivative of tanh

__kernel void array_dtanh_f32(__global float *a,
                              __global float *b) {
    uintptr_t i = get_global_id(0);
    // dtanh = 1 - tanh(x)^2
    b[i] = tanh(a[i]);
    b[i] = 1.0 - b[i]*b[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// sigmoid

__kernel void array_sigmoid_f32(__global float *a,
                                __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = sigmoid(a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// derivative of sigmoid

__kernel void array_dsigmoid_f32(__global float *a,
                                 __global float *b) {
    uintptr_t i = get_global_id(0);
    // dsigmoid = sigmoid(x)*(1 - sigmoid(x))
    b[i] = sigmoid(a[i]);
    b[i] = b[i]*(1.0 - b[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// log

__kernel void array_log_f32(__global float *a,
                            __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = log(a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// exp

__kernel void array_exp_f32(__global float *a,
                            __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = exp(a[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// negate

__kernel void array_negate_f32(__global float *a,
                               __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = -a[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rmsprop

__kernel void array_rmsprop_f32(__global float *x, __global float *dx, __global float *cache,
                                float learn_rate, float decay_rate, float eps) {
    uintptr_t i = get_global_id(0);
    cache[i] = decay_rate * cache[i] + (1.f - decay_rate) * dx[i]*dx[i];
    x[i] += learn_rate*dx[i] / (sqrt(cache[i]) + eps);
}
