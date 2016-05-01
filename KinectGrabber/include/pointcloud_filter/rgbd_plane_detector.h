/*
 * Created by sugar10w, 2016.3.19
 *
 * ��һ�ż�¼�����Ϣ��cv::Mat�ϲ��ұ�Ե��
 */

#ifndef _KINECT_DEPTH_CONTOUR__
#define _KINECT_DEPTH_CONTOUR__

#include <opencv2/opencv.hpp>
#include "common.h"

namespace tinker {
namespace vision {


/* ��pcl::PointXYZRGBNomal���ұ�Ե�� */
//cv::Mat GetEdgeFromDepth(PointCloudNTPtr & cloud);

/* Ϊ�˴���ǰ��ϸС����ĸ��Ŷ����������зֵ���������Զ����ڵ�������б����ʽ�����Ӳ����� */
//void ConnectDepthMask(PointCloudConstPtr & cloud, cv::Mat & mask);

/* �ӱ�Եͼ���ϣ�����ƽ���ȡ�ɰ� */
//cv::Mat GetMaskFromEdge(PointCloudConstPtr & cloud, PointCloudNTPtr & normals, cv::Mat & edge);



/* ���ڷ�����ȥ�������е�ƽ�� */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud);


}
}


#endif //_KINECT_DEPTH_CONTOUR__
