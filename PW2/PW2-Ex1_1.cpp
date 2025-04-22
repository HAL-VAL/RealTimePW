#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

// Anaglyph tipe
enum AnaglyphType {
    ORIGINAL = 0,
    TRUE_ANAGLYPH = 1,
    GRAY_ANAGLYPH = 2,
    COLOR_ANAGLYPH = 3,
    HALF_COLOR_ANAGLYPH = 4,
    OPTIMIZED_ANAGLYPH = 5
};

// default is ORIGINAL
int anaglyphType = 0;


Vec3b applyAnaglyph(int x, int y, const Mat& left, const Mat& right, int type) {
    Vec3b l = left.at<Vec3b>(y, x);
    Vec3b r = right.at<Vec3b>(y, x);

    float lGray = 0.299f * l[2] + 0.587f * l[1] + 0.114f * l[0];
    float rGray = 0.299f * r[2] + 0.587f * r[1] + 0.114f * r[0];

    switch (type) {
        case ORIGINAL:
            return l;
        case TRUE_ANAGLYPH:
            return Vec3b(rGray, 0, lGray);
        case GRAY_ANAGLYPH:
            return Vec3b(rGray, rGray, lGray);
        case COLOR_ANAGLYPH:
            return Vec3b(r[0], r[1], l[2]);
        case HALF_COLOR_ANAGLYPH:
            return Vec3b(r[0], r[1], lGray);
        case OPTIMIZED_ANAGLYPH:
            return Vec3b(r[0], r[1], static_cast<uchar>(0.7f * l[1] + 0.3f * l[0]));
        default:
            return Vec3b(255, 0, 255);
    }
}

void onTrackbar(int, void*) {
}


string getAnaglyphName(int type) {
    switch (type) {
        case 0: return "Original";
        case 1: return "True Anaglyph";
        case 2: return "Gray Anaglyph";
        case 3: return "Color Anaglyph";
        case 4: return "Half Color Anaglyph";
        case 5: return "Optimized Anaglyph";
        default: return "Unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file>" << endl; // San_Francisco.mp4
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    // make the right and left images
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int halfWidth = width / 2;

    namedWindow("Anaglyph Color Methods", WINDOW_AUTOSIZE);
    createTrackbar("Anaglyph Type", "Anaglyph Color Methods", NULL, 5, onTrackbar);

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        anaglyphType = getTrackbarPos("Anaglyph Type", "Anaglyph Color Methods");

        Mat left = frame(Rect(0, 0, halfWidth, height));
        Mat right = frame(Rect(halfWidth, 0, halfWidth, height));
        Mat result(height, halfWidth, CV_8UC3);

        // using OpenMP for using anaglyph
        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                result.at<Vec3b>(y, x) = applyAnaglyph(x, y, left, right, anaglyphType);
            }
        }

        Mat display;
        resize(result, display, Size(), 0.5, 0.5);  // reshape the window and video
        putText(display, getAnaglyphName(anaglyphType), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
        imshow("Anaglyph Color Methods", display);

        if (waitKey(30) == 27) break; // ESC key to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
