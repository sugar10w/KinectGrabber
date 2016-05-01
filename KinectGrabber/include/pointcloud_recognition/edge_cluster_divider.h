/* 
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.3.24
 *
 * 利用shard图层直接获取Contours。
 * 之后参考no_plane图层确定所有Contours是否为平面。
 */

#ifndef __EDGE_CLUSTER_DIVIDER_H__
#define __EDGE_CLUSTER_DIVIDER_H__

#include <opencv2/opencv.hpp>
#include "pointcloud_recognition/object_cluster.h"
#include "common.h"

namespace tinker {
namespace vision {

class EdgeClusterDivider
{
public:
    EdgeClusterDivider(PointCloudConstPtr & cloud, cv::Mat & shard_mask, cv::Mat & no_plane_mask);
    std::vector<ObjectCluster> GetDividedCluster();
    inline cv::Mat GetMask() { return mask_; }
    inline cv::Mat GetShardMask() { return shard_mask_; }
private:
    PointCloudConstPtr cloud_;
    cv::Mat shard_mask_;
    cv::Mat no_plane_mask_;
    cv::Mat mask_;

};

}
}    

#endif // __EDGE_CLUSTER_DIVIDER_H__

