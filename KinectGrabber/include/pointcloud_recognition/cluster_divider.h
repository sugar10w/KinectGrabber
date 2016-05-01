/*
 * Created by ����ة, 2016.1
 * Last edited by sugar10w, 2016.2.29
 *
 * ��һ�����Ʋ��Ϊ�������
 *
 */

#ifndef __CLUSTER_DIVIDER_H__
#define __CLUSTER_DIVIDER_H__

#include <opencv2/opencv.hpp>

#include "object_builder/object_cluster.h"
#include "common.h"

namespace tinker
{
namespace vision
{
class ClusterDivider
{
public:
    ClusterDivider(PointCloudPtr point_cloud);
    std::vector<ObjectCluster> GetDividedCluster();
private:
    PointCloudPtr point_cloud_;
    float leaf_size_;
};
}
}

#endif

