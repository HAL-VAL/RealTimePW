#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

// Calculate the determinant of the covariance matrix of a local RGB region
double calDeterminant (const Mat& region) {
    int n = region.rows * region.cols;
    if (n <= 1) return 0.0;

    // avg of each RGB
    Scalar avg = mean(region);
    Mat centered;
    region.convertTo(centered, CV_64F);

    // zero-mean
    #pragma omp parallel for
    for (int i = 0; i < centered.rows; ++i) {
        for (int j = 0; j < centered.cols; ++j) {
            Vec3d& px = centered.at<Vec3d>(i, j);
            px[0] -= avg[0];
            px[1] -= avg[1];
            px[2] -= avg[2];
        }
    }

    Mat data = centered.reshape(1, n); // reshape n x 3(RGB) matrix
    Mat cov, dummy;

    // Calculate the covariance matrix
    calcCovarMatrix(data, cov, dummy, COVAR_NORMAL | COVAR_ROWS);
    cov /= (n - 1); // noralization
    return determinant(cov);
}

// Gaussian Filter
Vec3b calcGaussian(const Mat& image, int x, int y, int regionSize, double factor) {
    // if regionSize = 5,
    int half = regionSize / 2; // = 5 / 2 = 2
    Rect roi(x - half, y - half, regionSize, regionSize); // Rect(3, 3, 5, 5)
    roi &= Rect(0, 0, image.cols, image.rows); // outside area becomes 0
    Mat neighborhood = image(roi);

    // Calculate the determinant of the covariance matrix of the neighborhood
    double det = calDeterminant (neighborhood);
    if (det < 1e-8) det = 1e-8; // minimum value to avoid division by zero

    // kernel size
    int radius = std::max(1, int(factor / det));
    if (radius > 10) radius = 10;
    int ksize = radius * 2 + 1;
    // sigma
    double sigma = ksize / 3.0;
    double twoSigma2 = 2 * sigma * sigma;

    // initialization
    Vec3d sum(0, 0, 0);
    double wsum = 0.0;

    // calculate Gaussian filter
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = clamp(y + dy, 0, image.rows - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = clamp(x + dx, 0, image.cols - 1);
            double weight = exp(-(dx * dx + dy * dy) / twoSigma2);
            Vec3b color = image.at<Vec3b>(yy, xx);
            sum += Vec3d(color) * weight;
            wsum += weight;
        }
    }

    sum /= wsum;
    return Vec3b(uchar(sum[0]), uchar(sum[1]), uchar(sum[2]));
}

int main(int argc, char** argv) {
    // input
    if (argc < 6) {
        cout << "Usage: " << argv[0] << " <input_image> <output_image> <neighborhood_size> <factor> <iterations>" << endl;
        return -1;
    }

    string inputPath = argv[1];
    string outputPath = argv[2];
    int neighborhood_size = atoi(argv[3]);
    double factor = atof(argv[4]);
    int iter = atoi(argv[5]); // number of iterations

    Mat input = imread(inputPath, IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Cannot read image: " << inputPath << endl;
        return -1;
    }

    Mat output;

    auto begin = chrono::high_resolution_clock::now();

    for (int i = 0; i < iter; ++i) {
        output = input.clone();

        #pragma omp parallel for
        for (int y = 0; y < input.rows; ++y) {
            for (int x = 0; x < input.cols; ++x) {
                output.at<Vec3b>(y, x) = calcGaussian(input, x, y, neighborhood_size, factor);
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - begin;

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    imwrite(outputPath, output);  // save the output image

    return 0;
}

