#include <torch/extension.h>
#include "max_dilated_conv2d.h"



at::Tensor max_dilated_conv2d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int im2col_step
){
    /**
    * get the input parameter's information
    **/
    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int height_out = (input_height  - (dilation_h * kernel_h)) / stride_h + 1;
    int width_out = (input_width - (dilation_w * kernel_w)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(in_channels % groups == 0, "in_channels must be divisible by group number");
    AT_ASSERTM(batch % im2col_step == 0, "batch must be divisible by im2col_step");
    AT_ASSERTM(kernel_channels == (in_channels / groups), "kernel_channels must be equal to in_channels / groups");
    AT_ASSERTM(out_channels % groups == 0, "out_channels must be divisible by group number");
    AT_ASSERTM(kernel_h % 2 == 1 || kernel_w % 2 ==1, "kernel_size must be odd number");
    /**
    * derive more information
    **/
    int kernel_dim = in_channels / groups * kernel_h * kernel_w;
    int step = std::min(im2col_step, batch);
    int input_dim = in_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;

    int M = out_channels / groups;
    int N = step * conv_out_spatial_dim;
    int K = kernel_dim;
    /**
    * malloc tmp space and output space
    **/
    auto col_buffer = at::empty({in_channels * kernel_h * kernel_w, step, conv_out_spatial_dim}, input.options());
    auto output = at::empty({batch, out_channels, height_out, width_out}, input.options());
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto weight_ptr = weight.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto output_ptr = output.data<float>();

    for (int n = 0; n < batch / step; ++n) {
        max_dilated_conv2d_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * step * input_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            step, height_out, width_out,
            col_buffer_ptr
        );
        for(int g = 0; g < groups; g++){
            auto output_instance_ptr = output_ptr + (n * groups * M  * N) + g * M * N;
            auto weight_tmp_ptr = weight_ptr + g * M * K;
            auto col_buffer_tmp_ptr = col_buffer_ptr + g * K * N;
            THCudaBlas_Sgemm(state, 'n', 'n', N, M, K, 1.0f, col_buffer_tmp_ptr, N, weight_tmp_ptr, K, 0.0f, output_instance_ptr, N);
   	 }
//   	 SwapAxis();
    }
    return output;
}

std::vector<at::Tensor> max_dilated_conv2d_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor out_grad,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int im2col_step
){
    /**
    * get the input parameter's information
    **/
    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int height_out = (input_height - (dilation_h * kernel_h)) / stride_h + 1;
    int width_out = (input_width - (dilation_w * kernel_w)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(in_channels % groups == 0, "in_channels must be divisible by group number");
    AT_ASSERTM(batch % im2col_step == 0, "batch must be divisible by im2col_step");
    AT_ASSERTM(kernel_channels == (in_channels / groups), "kernel_channels must be equal to in_channels / groups");
    AT_ASSERTM(out_channels % groups == 0, "out_channels must be divisible by group number");
    /**
    * derive more information
    **/
    int kernel_dim = in_channels / groups * kernel_h * kernel_w;
    int step = std::min(im2col_step, batch);
    int input_dim = in_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;

    int M = kernel_dim;
    int N = step * conv_out_spatial_dim;
    int K = out_channels / groups;
    /**
    * malloc tmp space and output space
    **/
    auto col_buffer = at::empty({in_channels * kernel_h * kernel_w, step, conv_out_spatial_dim}, input.options());
    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto weight_ptr = weight.data<float>();
    auto out_grad_ptr = out_grad.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto grad_input_ptr = grad_input.data<float>();
    auto grad_weight_ptr = grad_weight.data<float>();
    
    for (int n = 0; n < batch / step; ++n) {
        auto out_grad_instance_ptr = out_grad_ptr + n * groups * K * N;
        for(int g = 0;g < groups;g++){
            auto weight_tmp_ptr = weight_ptr + g * M * K;
            auto out_grad_instance_tmp_ptr = out_grad_instance_ptr + g * K * N;
            auto col_buffer_tmp_ptr = col_buffer_ptr + g * M * N;
            THCudaBlas_Sgemm(state,
            'n', 't',
            N, M, K,
            1.0f,
            out_grad_instance_tmp_ptr, N,
            weight_tmp_ptr, M,
            0.0f,
            col_buffer_tmp_ptr, N);
        }

        /**
        * calculate d loss / d input
        **/
        max_dilated_conv2d_col2im(
            THCState_getCurrentStream(state),
            col_buffer_ptr,
            input_ptr + n * step * input_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            step, height_out, width_out,
            grad_input_ptr + n * step * input_dim
        );

        /**
        * calculate d loss / d weight
        **/
        max_dilated_conv2d_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * step * input_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            step, height_out, width_out,
            col_buffer_ptr);
        for(int g = 0;g < groups;g++){
            auto grad_weight_tmp_ptr = grad_weight_ptr + g * M * K;
            auto out_grad_instance_tmp_ptr = out_grad_instance_ptr + g * K * N;
            auto col_buffer_tmp_ptr = col_buffer_ptr + g * M * N;
            THCudaBlas_Sgemm(state,
                't', 'n',
                M, K, N,
                1.0f,
                col_buffer_tmp_ptr, N,
                out_grad_instance_tmp_ptr, N,
                1.0f,
                grad_weight_tmp_ptr, M);
        }
    }
    return {grad_input, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &max_dilated_conv2d_forward, "max dilated conv2d forward (CUDA)");
  m.def("backward", &max_dilated_conv2d_backward, "max dilated dilated conv2d backward (CUDA)");
}
