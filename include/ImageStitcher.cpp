//
// Created by kon on 24-10-14.
//
#include "ImageStitcher.h"

void ImageStitcher::featherInit(cv::Mat imgOne, cv::Mat imgTwo) {
    cv::cvtColor(imgOne, imgOne, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgTwo, imgTwo, cv::COLOR_BGR2GRAY);

    akaze->detectAndCompute(imgOne, cv::noArray(), kp1, descriptors1);
    akaze->detectAndCompute(imgTwo, cv::noArray(), kp2, descriptors2);

    std::vector<std::vector<cv::DMatch> > kMatches;
    matcher->knnMatch(descriptors1, descriptors2, kMatches, 2);

    for (auto m: kMatches) {
        if (m[0].distance < 0.7 * m[1].distance)
            matches.push_back(m[0]);
    }
}

cv::Mat ImageStitcher::transform(cv::Mat imgOne, cv::Mat imgTwo) {
    if (matches.size() >= POINTS) {
        std::cout << "Find enough points to stitch" << std::endl;
        std::vector<cv::Point2f> srcPoints, dstPoints;
        for (auto m: matches) {
            srcPoints.push_back(kp1[m.queryIdx].pt);
            dstPoints.push_back(kp2[m.trainIdx].pt);
        }

        cv::Mat H = cv::findHomography(srcPoints, dstPoints, cv::RHO,2);
        float srcHeight = imgOne.rows;
        float srcWight = imgTwo.cols;
        std::vector<cv::Point2f> srcCorners(4);
        srcCorners[0] = cv::Point2f(0, 0);
        srcCorners[1] = cv::Point2f(srcWight - 1, 0);
        srcCorners[2] = cv::Point2f(srcWight - 1, srcHeight - 1);
        srcCorners[3] = cv::Point2f(0, srcHeight - 1);

        std::vector<cv::Point2f> transformCorners(4);
        try {
            cv::perspectiveTransform(srcCorners, transformCorners, H);
        } catch (...) {
            std::cout << "There is something wrong when doing points transform" << std::endl;
        }
        //计算变换后图片的坐标,进行平移,并且确定结果图片的大小
        std::vector<cv::Point2f> twoPoints(4);
        float twoHeight = imgTwo.rows;
        float twoWight = imgTwo.cols;
        twoPoints[0] = cv::Point2f(0, 0);
        twoPoints[1] = cv::Point2f(twoWight - 1, 0);
        twoPoints[2] = cv::Point2f(twoWight - 1, twoHeight - 1);
        twoPoints[3] = cv::Point2f(0, twoHeight - 1);

        std::vector<cv::Point2f> points_(transformCorners.begin(), transformCorners.end());
        points_.insert(points_.end(), twoPoints.begin(), twoPoints.end());
        auto [xMin_,xMax_] = std::minmax_element(points_.begin(), points_.end(),
                                                 [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });
        auto [yMin_,yMax_] = std::minmax_element(points_.begin(), points_.end(),
                                                 [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
        float xMin = xMin_->x;
        float xMax = xMax_->x;
        float yMin = yMin_->y;
        float yMax = yMax_->y;

        //手动构造平移矩阵
        cv::Mat M = (cv::Mat_<double>(3, 3) << 1, 0, -xMin, 0, 1, -yMin, 0, 0, 1);
        cv::Mat result(std::max(yMax - yMin + 1, -yMin + twoHeight + 1),
                       std::max(xMax - xMin + 1, -xMin + twoWight + 1), CV_16UC3);

        cv::warpPerspective(imgOne, result, M * H, result.size());
        cv::Rect2f roi(-xMin, -yMin, twoWight, twoHeight);
        imgTwo.copyTo(result(roi));

        return result;
    }else{
        std::cout << "No enough points to stitch" << std::endl;
        cv::Mat result;
        cv::hconcat(imgOne, imgTwo, result);

        return result;
    }
}

cv::Mat ImageStitcher::showFeatures(cv::Mat imgOne, cv::Mat imgTwo) {
    cv::Mat result;
    try {
        cv::drawMatches(imgOne, kp1, imgTwo, kp2, matches, result);
    }catch (...) {
        std::cout << "Failed to draw it !" << std::endl;
    }
    return result;
}

std::vector<cv::KeyPoint> ImageStitcher::getKeyPoints(int n) {
    if (n == 0) return kp1;
    return kp2;
}

std::vector<cv::DMatch> ImageStitcher::getMatches() {
    return matches;
}

cv::Mat ImageStitcher::getDescriptors(int n) {
    if(n==0) return descriptors1;
    return descriptors2;
}
