#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

// - detectors      ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
// - descriptors    ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]
// - matcherType    ["MAT_BF", "MAT_FLANN"]
// - descriptorType ["DES_BINARY", "DES_HOG"] --> depends on descriptor
// - selectorType   ["SEL_NN", "SEL_KNN"]
struct DetectionInfo {
    std::string detector, descriptor, descriptorType, matcherType, selectorType;
    std::vector<int> numKeypoints, numKeypointsVehicle, numKeypointsMatched; 
    std::vector<double> tKeypointDetection, tKeypointDescription;
};

#endif /* dataStructures_h */
