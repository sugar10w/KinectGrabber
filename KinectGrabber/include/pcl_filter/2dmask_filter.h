/*
 * Created by sugar10w, 2016.3.18
 * Last edited by sugar10w, 2016.3.18
 * 
 * ���ڴ�KinectSDK2��ȡ�����ݡ�
 * ��ȡ512*424�Ķ�ά�ɰ棬Ӧ�õ�ԭʼ�����ϡ�
 *
 */

#ifndef _POINTCLOUD_2DMASK_FILTER__
#define _POINTCLOUD_2DMASK_FILTER__

#include "common.h"
#include <opencv2/opencv.hpp>

namespace tinker {
namespace vision {

/*��ȡ512*424�Ķ�ά�ɰ棬Ӧ�õ�ԭʼ������*/
PointCloudPtr GetCloudFromMask(const cv::Mat & mask, const PointCloudConstPtr & cloud);    

/*��ȡ��Ч���ɰ�*/
cv::Mat GetValidMask(PointCloudConstPtr cloud);

/* ��ά���� */
void OpenImage(cv::Mat & img, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size);

/* ����ά�ĵ��ƽ��ж�ά���� */
void OpenImage(PointCloudPtr & cloud, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size);

/* ����Kinect�������Ե���ֲ���������ɢ�� */
cv::Mat GetShardMask(PointCloudConstPtr & cloud);

}    
}

#endif //_POINTCLOUD_2DMASK_FILTER__
