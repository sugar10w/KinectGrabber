/*
 * Created by sugar10w, 2016.3.19
 * Last edited by sugar10w, 2016.5.7
 *
 * �ӽṹ�����ƻ����ƽ���ɰ�
 */

#ifndef _KINECT_RGBD_PLANE_DETECTOR__
#define _KINECT_RGBD_PLANE_DETECTOR__

#include <opencv2/opencv.hpp>
#include "common.h"

namespace tinker {
namespace vision {

/* ���ڷ�����ȥ�������е�ƽ�� */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud);


}
}


#endif //_KINECT_RGBD_PLANE_DETECTOR__
