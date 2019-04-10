#ifndef MAX_DILATED_CONVOLUTION
#define MAX_DILATED_CONVOLUTION
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
extern THCState *state;
typedef std::vector<int> TShape;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}

inline TShape SubVector(const TShape &shape, int start, int end) {
    TShape res;
    for(int i=start;i<end;i++){
        res.push_back(shape[i]);
    }
    return res;
}

void max_dilated_conv2d_im2col(cudaStream_t stream,
    const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int height_out, const int width_out,
    float* data_col);


void max_dilated_conv2d_col2im(cudaStream_t stream,
    const float* data_col,
    const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int height_out, const int width_out,
    float* grad_im);

void SwapAxis();

#endif
