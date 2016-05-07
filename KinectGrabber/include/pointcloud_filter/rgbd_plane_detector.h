/*
 * Created by sugar10w, 2016.3.19
 * Last edited by sugar10w, 2016.5.7
 *
 * 从结构化点云获得无平面蒙版
 */

#ifndef _KINECT_RGBD_PLANE_DETECTOR__
#define _KINECT_RGBD_PLANE_DETECTOR__

#include <opencv2/opencv.hpp>
#include "common.h"

namespace tinker {
namespace vision {

/* 基于法向量去除点云中的平面 */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud);


}
}


#endif //_KINECT_RGBD_PLANE_DETECTOR__
