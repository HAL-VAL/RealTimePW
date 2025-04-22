#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

using namespace cv;

__global__
// Gaussian and determinant
void calcGaussianKernel(uchar3* input, uchar3* output, int width, int height, int pitch,
                             int regionSize, double factor) {
    // Block and thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto clamp = [](int v, int low, int high) {
        return max(low, min(v, high));
    };

    int half = regionSize / 2;
    double sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int xx = clamp(x + dx, 0, width - 1);
            int yy = clamp(y + dy, 0, height - 1);
            uchar3 color = input[yy * pitch + xx];
            sumB += color.x;
            sumG += color.y;
            sumR += color.z;
            count++;
        }
    }

    double avgR = sumR / count, avgG = sumG / count, avgB = sumB / count;
    double varRR = 0, varGG = 0, varBB = 0, varRG = 0, varRB = 0, varGB = 0;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int xx = clamp(x + dx, 0, width - 1);
            int yy = clamp(y + dy, 0, height - 1);
            uchar3 c = input[yy * pitch + xx];
            double r = c.z - avgR;
            double g = c.y - avgG;
            double b = c.x - avgB;
            varRR += r * r; varGG += g * g; varBB += b * b;
            varRG += r * g; varRB += r * b; varGB += g * b;
        }
    }

    varRR /= count; varGG /= count; varBB /= count;
    varRG /= count; varRB /= count; varGB /= count;

    double det = varRR * (varGG * varBB - varGB * varGB)
               - varRG * (varRG * varBB - varGB * varRB)
               + varRB * (varRG * varGB - varGG * varRB);

    if (det < 1e-8) det = 1e-8;
    int radius = max(1, int(factor / det));
    if (radius > 10) radius = 10;

    int ksize = 2 * radius + 1;
    double sigma = ksize / 3.0;
    double twoSigma2 = 2 * sigma * sigma;

    double sumRf = 0, sumGf = 0, sumBf = 0, wsum = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = clamp(x + dx, 0, width - 1);
            int yy = clamp(y + dy, 0, height - 1);
            uchar3 c = input[yy * pitch + xx];
            double w = exp(-(dx * dx + dy * dy) / twoSigma2);
            sumRf += c.z * w;
            sumGf += c.y * w;
            sumBf += c.x * w;
            wsum += w;
        }
    }

    uchar3 result;
    result.z = (uchar)(sumRf / wsum);
    result.y = (uchar)(sumGf / wsum);
    result.x = (uchar)(sumBf / wsum);
    output[y * pitch + x] = result;
}


void calcGaussianCUDA(const Mat& input, Mat& output, int regionSize, double factor, int iter,
                          int blockSizeX, int blockSizeY) {
    int width = input.cols, height = input.rows;
    size_t size = width * height * sizeof(uchar3);

    uchar3 *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);

    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((width + blockSizeX - 1) / blockSizeX, (height + blockSizeY - 1) / blockSizeY);

    for (int i = 0; i < iter; ++i) {
        calcGaussianKernel<<<grid, block>>>(d_input, d_output, width, height, width, regionSize, factor);
        cudaDeviceSynchronize();
        std::swap(d_input, d_output);
    }

    cudaMemcpy(output.ptr(), d_input, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
