/*
 * Created by sugar10w, 2016.3.19
 * Last edited by sugar10w, 2016.5.7 
 *
 * 从结构化点云获得无平面蒙版
 */

#include "pointcloud_filter/rgbd_plane_detector.h"

#include <cmath>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/features/integral_image_normal.h> 

#include "pointcloud_filter/2dmask_filter.h"


namespace tinker {
namespace vision {

static float square_distance(const float* n1, const float* n2)
{
    return 
        (n1[0]-n2[0])*(n1[0]-n2[0]) +
        (n1[1]-n2[1])*(n1[1]-n2[1]) +
        (n1[2]-n2[2])*(n1[2]-n2[2]);
}

/* 用pcl::PointXYZRGBNomal查找边缘点.
 * 此点的法向量方向, 与上下左右四个点的法向量差别过大的, 视作边缘点
 * TODO Kinect图像的**边缘处**的畸变非常大, 法向量变化量基本都会超过现在设定的阈值, 导致边缘部分的平面不能正常识别! */
static cv::Mat GetEdgeFromDepth(PointCloudNTPtr & cloud)
{
    cv::Mat edge(cloud->height, cloud->width, CV_8UC1);
    
    /* 注意ij的方向 */
    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j, ++k)
        {
            /* 最边上的点不考虑 */
            if (j==0 || j+1==cloud->width 
                || i==0 || i+1==cloud->height)
            {
                edge.at<uchar>(i, j)=0;
                continue;
            }
            
            /* 提取上下左右的法向量 */
            const PointNT 
                & pt  = cloud->points[k],
                & pt1 = cloud->points[k-cloud->width],
                & pt2 = cloud->points[k+cloud->width],
                & pt3 = cloud->points[k-1],
                & pt4 = cloud->points[k+1];
            const float *nn  = pt.normal,
                        *nn1 = pt1.normal,
                        *nn2 = pt2.normal,
                        *nn3 = pt3.normal,
                        *nn4 = pt4.normal;            
            /* 法向量判定 */
            if ( square_distance(nn, nn1) + square_distance(nn, nn2) +
                 square_distance(nn, nn3) + square_distance(nn, nn4) > 0.03f)
                edge.at<uchar>(i, j) = 255;
            else 
                edge.at<uchar>(i, j) = 0;
        }
    return edge;
}

/* 将面积太小的黑色块填充起来 */
static void RoughFixMask(PointCloudConstPtr &cloud, cv::Mat & mask)
{
    /* 获取轮廓 */
    const cv::Mat contour = ~mask;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    /* 把面积小的填起来 */
    for (int idx=0; idx<contours.size(); ++idx)
    {
        if (contours[idx].size()>1800) continue;
        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size<300) 
            cv::drawContours(mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
    }
}

/* 获取pt1-->pt2向量, 默认进行向量单位化. */
static pcl::PointXYZ GetVector(PointT pt1, PointT pt2, bool unit=true)
{
    pcl::PointXYZ vector;
    vector.x = pt2.x - pt1.x;
    vector.y = pt2.y - pt1.y;
    vector.z = pt2.z - pt1.z;
    if (!unit) return vector;

    float m = sqrt(vector.x*vector.x+vector.y*vector.y+vector.z*vector.z);
    vector.x /= m;
    vector.y /= m;
    vector.z /= m;
    return vector;
}
/* 计算vector点到unit_vector所在直线的距离 */
static float CrossDistance(pcl::PointXYZ unit_vector, pcl::PointXYZ vector)
{
    pcl::PointXYZ result;
    result.x = unit_vector.y * vector.z - unit_vector.z * vector.y;
    result.y = unit_vector.z * vector.x - unit_vector.x * vector.z;
    result.z = unit_vector.x * vector.y - unit_vector.y * vector.x;
    return sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
}

/* 为了处理前端细小物体的干扰而导致区域被切分的情况，尝试对相邻的区域进行辨别形式的连接操作。 */
static void ConnectDepthMask(PointCloudConstPtr & cloud, cv::Mat & mask)
{
    /* 水平方向抽取一部分的线进行处理。注意, 此时黑色是有效点。 */
    const int step = 5;
    /* 直线判定时, 可接受的最大偏离误差 */
    const float dist_limit = 0.02f;

    for (int x=step; x<cloud->height-1; x+=step)
    {
        /* 水平抽取连续黑色点, 并判定他们是否可能属于某个大平面 */
        std::vector<cv::Vec2i> pair_list;
        for (int y=0; y<cloud->width; ++y)
        {
            if (mask.at<uchar>(x,y)) continue;        
            
            int start = y, end = y+1;
            while ( end<cloud->width && !mask.at<uchar>(x, end)) ++end;
            if (end>=cloud->width) break;
            --end;
            /* 先用线段长度初步判断 */
            if (start==0 || end-start<=8) { y=end+1; continue;}

            /* 仔细检查是否共线 */
            bool valid = true;
            /* 获取首尾连线的方程 */
            const PointT & start_point = cloud->points[(x+1)*cloud->width-start-1];
            pcl::PointXYZ unit_vector = GetVector(start_point, cloud->points[(x+1)*cloud->width-end-1]);
            /* 逐个点判断是否超标 */
            for (int i=start+1; i<end; ++i)
                if (CrossDistance(unit_vector, 
                            GetVector(start_point, cloud->points[(x+1)*cloud->width-i-1], false)) > dist_limit)
                {
                    valid = false;
                    break;
                }
            if (valid) pair_list.push_back(cv::Vec2i(start, end));
            
            /* 注意y的跳跃变化 */
            y = end;
        }

        /* 测试相邻长线段, 并尝试连接 */
        for (int i=1; i<pair_list.size(); ++i)
        {
            bool bind = true;
            /* 获取左长线段的方程 */
            PointT start_point = cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1];
            pcl::PointXYZ unit_vector = GetVector(
                cloud->points[(x+1)*cloud->width-pair_list[i-1][0]-1],
                cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1]);
            /* 用左长线段方程判定右长线段 */
            for (int j=pair_list[i][0]; j<pair_list[i][1]; ++j)
                if (CrossDistance(unit_vector,
                            GetVector(start_point, cloud->points[(x+1)*cloud->width-j-1], false)) > dist_limit)
                {
                    bind = false;
                    break;
                }

            /* 确定左右长线段是共线的, 在蒙版上连接之 */
            if (bind)
            {
                /* 绘制三条线，防止误连 TODO */
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x-1,j)=255;
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x,j)=0;
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x+1,j)=255;
            }
        }
    }

}

/* 从边缘图像上，过滤平面获取蒙版 */
static cv::Mat GetMaskFromEdge(PointCloudConstPtr & cloud, PointCloudNTPtr & normals, cv::Mat & edge)
{
    cv::Mat output = edge.clone();

    /* 获取edge的轮廓 */
    const cv::Mat contour = ~ edge;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    /* 逐个精细判断 */
    for (int idx=0; idx<contours.size(); ++idx)
    {
        /* 扫描每一个点 */
        cv::Mat inside(edge.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(inside, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        float nx_min = 2, nx_max = -2,
              ny_min = 2, ny_max = -2,
              nz_min = 2, nz_max = -2,
              x_min = 10000, x_max = -10000,
              y_min = 10000, y_max = -10000,
              z_min = 10000, z_max = -10000;
        /* 注意ij的方向 */
        int k = 0;
        for (int i=0; i<cloud->height; ++i)
            for (int j=cloud->width-1; j>=0; --j, ++k)
            {
                if (inside.at<uchar>(i,j))
                {
                    const PointT  & pt  = cloud->points[k];
                    PointNT & pnt = normals->points[k];

                    if (pt.x<x_min) x_min = pt.x;
                    if (pt.x>x_max) x_max = pt.x;
                    if (pt.y<y_min) y_min = pt.y;
                    if (pt.y>y_max) y_max = pt.y;
                    if (pt.z<z_min) z_min = pt.z;
                    if (pt.z>z_max) z_max = pt.z;
                    if (pnt.normal_x<nx_min) nx_min = pnt.normal_x;
                    if (pnt.normal_x>nx_max) nx_max = pnt.normal_x;
                    if (pnt.normal_y<ny_min) ny_min = pnt.normal_y;
                    if (pnt.normal_y>ny_max) ny_max = pnt.normal_y;
                    if (pnt.normal_z<nz_min) nz_min = pnt.normal_z;
                    if (pnt.normal_z>nz_max) nz_max = pnt.normal_z;                    
                }
            }
        /* 通过判定尺寸, 法向量偏转, 确定平面 */
        if (! ( (x_max-x_min)>0.6
            || (y_max-y_min)>0.6
            || (z_max-z_min)>0.6
            || ( (nx_max-nx_min)<0.01 && (ny_max-ny_min)<0.01 && (nz_max-nx_min)<0.01 ))) //TODO
        {
            cv::drawContours(output, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        }
    }

    return  output;
}

/* 基于法向量去除点云中的平面 */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud)
{
    /* 声明法向量 */
    PointCloudNTPtr normals(new pcl::PointCloud<PointNT>);
    normals->width = cloud->width; 
    normals->height = cloud->height;
    normals->resize(normals->width*normals->height);
    /* 计算法向量 */
    pcl::IntegralImageNormalEstimation<PointT, PointNT> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE );
    ne.setDepthDependentSmoothing(false);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    /* 利用法向量信息获取边缘(非平面部分) */
    cv::Mat edge = GetEdgeFromDepth(normals);
    /* 白色部分扩张1 */
    OpenImage(edge, 1, 0, 0);
    
    /* 先尝试修复过小的黑色块 */
    RoughFixMask(cloud, edge);
    /* 尝试黑色块平面连接 */
    ConnectDepthMask(cloud, edge);

    /* 通过尺寸和法向量获取最终的蒙版 */
    cv::Mat no_plane_mask = GetMaskFromEdge(cloud, normals, edge);
    /* 修复ConnectDepth; 修整蒙版 */
    OpenImage(no_plane_mask, 1, 2, 0);

    return no_plane_mask;
}



}
}

