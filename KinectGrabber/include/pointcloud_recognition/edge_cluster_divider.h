/* 
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.5.7
 *
 * 利用shard图层获取轮廓.
 * 之后参考no_plane图层, 将非平面的轮廓视为目标.
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
    void GetDividedCluster(std::vector<ObjectCluster>& divided_objects);
    inline cv::Mat GetMask() { return mask_; }
private:
    PointCloudConstPtr cloud_;
    const cv::Mat & shard_mask_;
    const cv::Mat & no_plane_mask_;
    cv::Mat mask_;

};

}
}    

#endif // __EDGE_CLUSTER_DIVIDER_H__

