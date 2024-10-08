import paddle
import paddle.nn.functional as F

def gaussian_blur(input, kernel_size, sigma):
    # Ensure input is a paddle.Tensor
    if not isinstance(input, paddle.Tensor):
        raise TypeError(f"Expected input to be a paddle.Tensor, got: {type(input)}")

    def get_gaussian_kernel(kernel_size, sigma):
        ax = paddle.arange(kernel_size, dtype=paddle.float32) - (kernel_size - 1) / 2.0
        xx, yy = paddle.meshgrid(ax, ax)
        kernel = paddle.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()  # Normalize
        return kernel

    channels = input.shape[1]
    if isinstance(kernel_size, (tuple, list)):
        kernel_size = kernel_size[0]  # Assuming square kernel
    if isinstance(sigma, (tuple, list)):
        sigma = sigma[0]  # Assuming same sigma for all channels

    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma)

    # Expand kernel to match the input channels
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.tile([channels, 1, 1, 1])  # Shape: (channels, 1, kernel_size, kernel_size)

    # Perform convolution using paddle.nn.functional
    output = F.conv2d(input, gaussian_kernel, padding=kernel_size // 2, groups=channels)

    return output

# Example usage:
# Make sure input_image is a paddle.Tensor of shape (N, C, H, W)
# blurred_image = gaussian_blur(input_image, (5, 5), (1.0, 1.0))
