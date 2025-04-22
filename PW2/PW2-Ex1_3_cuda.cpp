#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>


using namespace std;
using namespace cv;

// Host function prototypes for CUDA kernels
void calcGaussianCUDA(const Mat& input, Mat& output, int regionSize, double factor, int iter,
                          int blockSizeX, int blockSizeY);

// check the benchmark
void benchmarkCudaKernel(const Mat& input, Mat& output, int regionSize, double factor, int iter) {
    vector<int> block_sizes = {8, 16, 32};

    for (int bsize : block_sizes) {
        dim3 block(bsize, bsize);
        cout << "\nBlock size: " << bsize << "x" << bsize << endl;

        auto start = chrono::high_resolution_clock::now();
        calcGaussianCUDA(input, output, regionSize, factor, iter, bsize, bsize);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> duration = end - start;
        double total = duration.count();
        double per_iter = total / iter;

        cout << "Total time: " << total << " s" << endl;
        cout << "Time per iteration: " << per_iter << " s" << endl;
        cout << "IPS: " << iter / total << endl;
    }
}

int main(int argc, char** argv) {
    // input
    if (argc < 6) {
        cout << "Usage: " << argv[0] << " <input> <output> <neighborhood_size> <factor> <iterations>\n";
        return -1;
    }

    string inputPath = argv[1];
    string outputPath = argv[2];
    int regionSize = atoi(argv[3]);
    double factor = atof(argv[4]);
    int iter = atoi(argv[5]);

    Mat input = imread(inputPath, IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Failed to load image: " << inputPath << endl;
        return -1;
    }

    Mat output = input.clone();

    benchmarkCudaKernel(input, output, regionSize, factor, iter);
    imwrite(outputPath, output);

    return 0;
}
