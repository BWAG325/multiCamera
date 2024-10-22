//
// Created by kon on 24-10-14.
//
#include "include/ImageStitcher.h"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <chrono>


# define  WIDTH 1280
# define  HEIGHT 720

// # define  WIDTH 720
// # define  HEIGHT 480

std::shared_mutex rwMutex;
std::atomic<bool> stopFlag(false);
std::atomic<bool> showFps(false);

void cameraSet(cv::VideoCapture &cap) {
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // cap.set(cv::CAP_PROP_EXPOSURE, 2000);

    // 设置分辨率到720p (1280x720)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    // 设置帧率到60fps
    cap.set(cv::CAP_PROP_FPS, 60);
    // cap.set(cv::CAP_PROP_BUFFERSIZE, 30);

    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps_ = cap.get(cv::CAP_PROP_FPS);
    double exposure_time = cap.get(cv::CAP_PROP_EXPOSURE);

    std::cout << "Camera resolution set to: " << width << "x" << height << std::endl;
    std::cout << "Camera frame rate set to: " << fps_ << "fps" << std::endl;
    std::cout << "Camera exposure time set to: " << exposure_time << "ms" << std::endl;
}

void preTransform(cv::Mat src,cv::Mat &result) {
    double alpha = 60.0;
    double w = src.cols;
    double h = src.rows;
    double f = w*2/tan(alpha/180*M_PI);

    int maxCol = f * atan((w-1 - w / 2) / f) + f * atan(w / (2.0 * f)) + 0.5;
    cv::Mat dis = cv::Mat::zeros(src.rows, maxCol, src.type());
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double x = j;
            double y = i;
            double x1 = f * atan((x - w / 2) / f) + f * atan(w / (2.0 * f));
            double y1 = f * (y - h / 2.0 ) / sqrtf(powf(x-w/2.0,2)+powf(f,2))+h/2.0;

            int col = int(x1+0.5);
            int row = int(y1+0.5);

            if (col <= src.cols && row <= src.rows && col >= 0 && row >= 0) {
                maxCol = std::max(col,maxCol);
                dis.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(i, j);
            }
        }
    }
    result = dis;
}


void cameraGet(cv::VideoCapture &cap, cv::Mat &frame) {
    using namespace std::chrono_literals;
    auto start = std::chrono::high_resolution_clock::now();
    auto lastStart = start;
    while (!stopFlag) {
        {
            std::unique_lock<std::shared_mutex> lock(rwMutex);
            cap >> frame;
            if (frame.empty()) {
                std::cout << "[Thread] empty frame" << std::endl;
            }
        }
        lastStart = start;
        start = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = start - lastStart;
        int fps = 1s / elapsed_time;
        preTransform(frame,frame);
        // std::this_thread::sleep_for(1ms);
        if(showFps)
            cv::putText(frame, std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    cv::Mat frameOne_,frameTwo_,frameThree_;
    cv::VideoCapture capOne(0,cv::CAP_V4L2);
    cv::VideoCapture capTwo(2,cv::CAP_V4L2);
    // cv::VideoCapture capThree(2,cv::CAP_V4L2);
    cameraSet(capOne);
    cameraSet(capTwo);
    // cameraSet(capThree_);
    std::thread threadOne(cameraGet, std::ref(capOne), std::ref(frameOne_));
    std::thread threadTwo(cameraGet, std::ref(capTwo), std::ref(frameTwo_));
    // std::thread threadThree(cameraGet, std::ref(capThree), std::ref(frameThree_));

    cv::namedWindow("Left", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Main", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Right", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("result",cv::WINDOW_NORMAL);
    cv::Mat frameOne,frameTwo,frameThree;

    auto start = std::chrono::high_resolution_clock::now();
    auto lastStart = start;

    while(true) {
        using namespace std::chrono_literals;
        lastStart = start;
        start = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = start - lastStart;
        int fps = static_cast<int>(1.0 / elapsed_time.count());
        std::cout << fps << std::endl;
        char getChar = cv::waitKey(1);
        {
            frameOne = frameOne_.clone();
            frameTwo = frameTwo_.clone();
            // frameThree = frameThree_;
        }
        if(frameOne.empty() || frameTwo.empty()) {
            std::cout << "[Main] empty frame" << std::endl;
            continue;
        }
        // frameThree = preTransform(frame_three);
        cv::imshow("Left", frameOne);
        cv::imshow("Main", frameTwo);
        // cv::imshow("Right", frameThree);

        ImageStitcher stitcher;
        std::cout << "Start stitching" << std::endl;
        stitcher.featherInit(frameOne, frameTwo);
        cv::Mat result = stitcher.transform(frameOne, frameTwo);
        // stitcher.featherInit(frameThree,mid);
        // cv::Mat result = stitcher.transform(frameThree,mid);
        cv::Mat feature = stitcher.showFeatures(frameOne, frameTwo);

        // cv::imshow("result", mid);
        cv::imshow("feature", feature);
        cv::imshow("result", result);

        if (getChar == 'q') {
            stopFlag = true;
            break;
        }else if (getChar == 'f') {
            showFps = !showFps;
        }
    }
    threadOne.join();
    threadTwo.join();
    capOne.release();
    capTwo.release();
    // capThree.release();
    cv::destroyAllWindows();

    return 0;
}