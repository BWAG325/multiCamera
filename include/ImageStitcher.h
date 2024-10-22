//
// Created by kon on 24-10-14.
//

#ifndef IMAGESTITCHER_H
#define IMAGESTITCHER_H

#include <opencv2/opencv.hpp>
#define POINTS 40

class ImageStitcher {
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat descriptors1, descriptors2;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING2);
    std::vector<cv::DMatch> matches;
public:
    void featherInit(cv::Mat imgOne, cv::Mat imgTwo);

    cv::Mat transform(cv::Mat imgOne, cv::Mat imgTwo);

    cv::Mat showFeatures(cv::Mat imgOne,cv::Mat imgTwo);

    std::vector<cv::KeyPoint> getKeyPoints(int n);
    std::vector<cv::DMatch> getMatches();
    cv::Mat getDescriptors(int n);
};

#endif