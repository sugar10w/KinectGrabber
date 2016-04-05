/*
 * Created by sugar10w, 2016.2.25
 * Last edited by sugar10w, 2016.2.25
 *
 * TODO ��¼������������ĵ���
 *
 */

#ifndef __OBJECTFINDER_OBJECT_CLUSTER_H__
#define __OBJECTFINDER_OBJECT_CLUSTER_H__


#include <pcl/visualization/pcl_visualizer.h>

#include "common.h"

namespace tinker {
namespace vision {

class ObjectCluster
{
public:
    /* �½�����*/
    ObjectCluster(const PointCloudPtr& input_cloud, float leaf_size=0.003);
    /* ��viewer�ϻ��Ʒ��� */
    void DrawBoundingBox(pcl::visualization::PCLVisualizer& viewer, int object_number);
    inline bool isValid() { return valid_; }
private:
    float leaf_size_;
    /* ��ŵ�λ�ú���ɫ��Ϣ*/
    float x_min, x_max, y_min, y_max, z_min, z_max, xx, yy, zz;
    int r_avg, g_avg, b_avg;
    /* ����*/
    PointCloudPtr cloud;
    
    bool valid_;

};

}
}

#endif //OBJECTFINDER_OBJECT_CLUSTER_H__
