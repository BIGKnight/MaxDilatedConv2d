#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "max_dilated_conv2d.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float get_data(    
    const float* data,
    const int data_width,
    const int height,
    const int width,
    const int h,
    const int w){
    if (h < 0 || w < 0 || h > height -1 || w > width - 1)
        return 0;
    else{
        return data[h * data_width + w];
    }
}    

__global__ void max_dilated_conv2d_im2col_kernel(
    int n,
    const float* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int num_channels,
    const int height_col, const int width_col,
    float* data_col
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % im2col_step;
        const int c_im = (index / width_col / height_col) / im2col_step;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h;
        const int w_in = w_col * stride_w;

        float* data_col_ptr = data_col + ((c_col * im2col_step + b_col) * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
        
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                
                float max_value = get_data(data_im_ptr, width, height, width, h_im, w_im);
                
                float tmp_value = static_cast<float>(0);
                for(int l = 0;l<dilation_h;++l){
                    for (int k = 0; k < dilation_w; ++k) {
                        int cur_h = h_im + l;
                        int cur_w = w_im + k;
                        tmp_value = get_data(data_im_ptr, width, height, width, cur_h, cur_w);
                        if (tmp_value > max_value)
                            max_value = tmp_value; 
                    }
                }
                
                *data_col_ptr = max_value;
                data_col_ptr += im2col_step * height_col * width_col;
            }
        }
    }
}


__global__ void max_dilated_conv2d_col2im_kernel(
    const int n,
    const float* data_col,
    const float* data_im,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int height_col, const int width_col,
    float* grad_im
){
    CUDA_KERNEL_LOOP(index, n){
        // the relative location in the filter
        const int j = (index / width_col / height_col / im2col_step) % kernel_w;
        const int i = (index / width_col / height_col / im2col_step / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / im2col_step / kernel_w / kernel_h; // which channel
        // 计算当前这个index对应的值被卷积操作的哪个内积点(也就是输出的spatial location)使用了.
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        // which step
        int b = (index / width_col / height_col) % im2col_step;
        
        const float* data_im_ptr = data_im + (b * channels + c) * height * width;
        
        const int h_in = h_out * stride_h ;
        const int w_in = w_out * stride_w ;
        
        const int cur_inv_h_grid = h_in + i * dilation_h;
        const int cur_inv_w_grid = w_in + j * dilation_w;
        
        float max_value = get_data(data_im_ptr, width, height, width, cur_inv_h_grid, cur_inv_w_grid);
        float tmp_value = static_cast<float>(0);
        int max_h = cur_inv_h_grid;
        int max_w = cur_inv_w_grid;
        
        for(int l = 0;l<dilation_h;++l){
            for (int k = 0; k < dilation_w; ++k) {
                        int cur_h = cur_inv_h_grid + l;
                        int cur_w = cur_inv_w_grid + k;
                        tmp_value = get_data(data_im_ptr, width, height, width, cur_h, cur_w);
                        if (tmp_value > max_value){
                            max_value = tmp_value; 
                            max_h = cur_h;
                            max_w = cur_w;
                        }
                    }
                }
        
        const float cur_top_grad = data_col[index];
        int cur_bottom_grad_pos = ((b * channels + c) * height + max_h) * width + max_w;
        atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);
    }
}


void max_dilated_conv2d_im2col(cudaStream_t stream,
    const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int height_out, const int width_out,
    float* data_col){
    int num_kernels = in_channels * im2col_step * height_out * width_out;
    max_dilated_conv2d_im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            im2col_step, in_channels,
            height_out, width_out,
            data_col
    );
}


void max_dilated_conv2d_col2im(cudaStream_t stream,
    const float* data_col, const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int im2col_step, const int height_out, const int width_out,
    float* grad_im){
    int  num_kernels = in_channels * kernel_h * kernel_w * im2col_step * height_out * width_out;
    max_dilated_conv2d_col2im_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_col,
        data_im,
        in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        im2col_step, height_out, width_out,
        grad_im
    );
}
