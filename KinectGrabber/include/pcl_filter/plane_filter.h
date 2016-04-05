/*
 * Created by sugar10w, 2016.2.27 
 * Last edited by sugar10w, 2016.2.28
 *
 * 找到点云中的平面
 * 从其他点云中去除平面
 *
 */

#ifndef __PCL_PLANE_FILTER__
#define __PCL_PLANE_FILTER__

#include<opencv2/opencv.hpp>
#include"common.h"
#include<pcl/ModelCoefficients.h>

namespace tinker {
namespace vision {

class PlaneFilter
{
public:
    /* cloud 将用于提取平面的点云
     * leaf_size 降采样尺寸 */
    PlaneFilter(PointCloudPtr &cloud, float leaf_size=0.1f);
    PlaneFilter(std::vector<pcl::ModelCoefficients> planes);
    ~PlaneFilter();

    /* 获取过滤蒙版 */
    cv::Mat GetMask(PointCloudConstPtr &cloud, float remove_radius=0.02f);

    /* cloud 将在cloud中去除平面 */
    void Filter(PointCloudPtr &cloud, float remove_radius=0.05f);

    /* 确定是否已经找到平面 */
    bool PlaneFound();

private:
    float leaf_size_;
    //Eigen::VectorXf plane_;
    std::vector<pcl::ModelCoefficients> planes_;
    bool plane_found_; 
};    

}
}

#endif //__PCL_PLANE_FILTER__
