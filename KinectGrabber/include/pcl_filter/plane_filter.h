/*
 * Created by sugar10w, 2016.2.27 
 * Last edited by sugar10w, 2016.2.28
 *
 * �ҵ������е�ƽ��
 * ������������ȥ��ƽ��
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
    /* cloud ��������ȡƽ��ĵ���
     * leaf_size �������ߴ� */
    PlaneFilter(PointCloudPtr &cloud, float leaf_size=0.1f);
    PlaneFilter(std::vector<pcl::ModelCoefficients> planes);
    ~PlaneFilter();

    /* ��ȡ�����ɰ� */
    cv::Mat GetMask(PointCloudConstPtr &cloud, float remove_radius=0.02f);

    /* cloud ����cloud��ȥ��ƽ�� */
    void Filter(PointCloudPtr &cloud, float remove_radius=0.05f);

    /* ȷ���Ƿ��Ѿ��ҵ�ƽ�� */
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
