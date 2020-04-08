# Sensor Fusion Nanodegree: Camera 2D Feature Tracking
Part of Sensor Fusion Nanodegree at Udacity

## Installation and Setup
For solving all given tasks, OpenCV 4.1 and its contributions (used for SIFT, BRIK - detectors) are needed. I installed OpenCV and OpenCV_Contrib via GIT and CMake (https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html). In Windows, the path to OpenCV needs to be added to system PATH-variable. Furthermore, an own OPENCV_DIR variable needs to be set. 

## Description of the Project Structure
* Segmentation
  * /src/quiz/ransac
* Clustering
  * /src/quiz/cluster
* Main project
  * /src
  
## Project Description

### MP.1 Use ring buffer
```c++
// if dataBuffer has reached its maximum size before appending new image,
// delete first element
if (dataBuffer.size() == dataBufferSize) 
    dataBuffer.erase(dataBuffer.begin());        

// push image into data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
``` 

### MP.2 Add more keypoint detectors
See code for more information. 

### MP.3 Limit keypoint area to rectangle in front
```c++
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
// cv::Rect = (x,y, width, height)
cv::Rect vehicleRect(535, 180, 180, 150);       
if (bFocusOnVehicle)
{
    std::vector<cv::KeyPoint> keypointsRect;
    for (auto i : keypoints) {                
        if (vehicleRect.contains(i.pt))
        {
            keypointsRect.push_back(i);
        }
    }
    keypoints = keypointsRect;
}   
```

### MP.4
