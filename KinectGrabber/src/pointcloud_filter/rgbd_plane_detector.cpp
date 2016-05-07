/*
 * Created by sugar10w, 2016.3.19
 * Last edited by sugar10w, 2016.5.7 
 *
 * �ӽṹ�����ƻ����ƽ���ɰ�
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

/* ��pcl::PointXYZRGBNomal���ұ�Ե��.
 * �˵�ķ���������, �����������ĸ���ķ������������, ������Ե��
 * TODO Kinectͼ���**��Ե��**�Ļ���ǳ���, �������仯���������ᳬ�������趨����ֵ, ���±�Ե���ֵ�ƽ�治������ʶ��! */
static cv::Mat GetEdgeFromDepth(PointCloudNTPtr & cloud)
{
    cv::Mat edge(cloud->height, cloud->width, CV_8UC1);
    
    /* ע��ij�ķ��� */
    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j, ++k)
        {
            /* ����ϵĵ㲻���� */
            if (j==0 || j+1==cloud->width 
                || i==0 || i+1==cloud->height)
            {
                edge.at<uchar>(i, j)=0;
                continue;
            }
            
            /* ��ȡ�������ҵķ����� */
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
            /* �������ж� */
            if ( square_distance(nn, nn1) + square_distance(nn, nn2) +
                 square_distance(nn, nn3) + square_distance(nn, nn4) > 0.03f)
                edge.at<uchar>(i, j) = 255;
            else 
                edge.at<uchar>(i, j) = 0;
        }
    return edge;
}

/* �����̫С�ĺ�ɫ��������� */
static void RoughFixMask(PointCloudConstPtr &cloud, cv::Mat & mask)
{
    /* ��ȡ���� */
    const cv::Mat contour = ~mask;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    /* �����С�������� */
    for (int idx=0; idx<contours.size(); ++idx)
    {
        if (contours[idx].size()>1800) continue;
        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size<300) 
            cv::drawContours(mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
    }
}

/* ��ȡpt1-->pt2����, Ĭ�Ͻ���������λ��. */
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
/* ����vector�㵽unit_vector����ֱ�ߵľ��� */
static float CrossDistance(pcl::PointXYZ unit_vector, pcl::PointXYZ vector)
{
    pcl::PointXYZ result;
    result.x = unit_vector.y * vector.z - unit_vector.z * vector.y;
    result.y = unit_vector.z * vector.x - unit_vector.x * vector.z;
    result.z = unit_vector.x * vector.y - unit_vector.y * vector.x;
    return sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
}

/* Ϊ�˴���ǰ��ϸС����ĸ��Ŷ����������зֵ���������Զ����ڵ�������б����ʽ�����Ӳ����� */
static void ConnectDepthMask(PointCloudConstPtr & cloud, cv::Mat & mask)
{
    /* ˮƽ�����ȡһ���ֵ��߽��д���ע��, ��ʱ��ɫ����Ч�㡣 */
    const int step = 5;
    /* ֱ���ж�ʱ, �ɽ��ܵ����ƫ����� */
    const float dist_limit = 0.02f;

    for (int x=step; x<cloud->height-1; x+=step)
    {
        /* ˮƽ��ȡ������ɫ��, ���ж������Ƿ��������ĳ����ƽ�� */
        std::vector<cv::Vec2i> pair_list;
        for (int y=0; y<cloud->width; ++y)
        {
            if (mask.at<uchar>(x,y)) continue;        
            
            int start = y, end = y+1;
            while ( end<cloud->width && !mask.at<uchar>(x, end)) ++end;
            if (end>=cloud->width) break;
            --end;
            /* �����߶γ��ȳ����ж� */
            if (start==0 || end-start<=8) { y=end+1; continue;}

            /* ��ϸ����Ƿ��� */
            bool valid = true;
            /* ��ȡ��β���ߵķ��� */
            const PointT & start_point = cloud->points[(x+1)*cloud->width-start-1];
            pcl::PointXYZ unit_vector = GetVector(start_point, cloud->points[(x+1)*cloud->width-end-1]);
            /* ������ж��Ƿ񳬱� */
            for (int i=start+1; i<end; ++i)
                if (CrossDistance(unit_vector, 
                            GetVector(start_point, cloud->points[(x+1)*cloud->width-i-1], false)) > dist_limit)
                {
                    valid = false;
                    break;
                }
            if (valid) pair_list.push_back(cv::Vec2i(start, end));
            
            /* ע��y����Ծ�仯 */
            y = end;
        }

        /* �������ڳ��߶�, ���������� */
        for (int i=1; i<pair_list.size(); ++i)
        {
            bool bind = true;
            /* ��ȡ���߶εķ��� */
            PointT start_point = cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1];
            pcl::PointXYZ unit_vector = GetVector(
                cloud->points[(x+1)*cloud->width-pair_list[i-1][0]-1],
                cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1]);
            /* �����߶η����ж��ҳ��߶� */
            for (int j=pair_list[i][0]; j<pair_list[i][1]; ++j)
                if (CrossDistance(unit_vector,
                            GetVector(start_point, cloud->points[(x+1)*cloud->width-j-1], false)) > dist_limit)
                {
                    bind = false;
                    break;
                }

            /* ȷ�����ҳ��߶��ǹ��ߵ�, ���ɰ�������֮ */
            if (bind)
            {
                /* ���������ߣ���ֹ���� TODO */
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x-1,j)=255;
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x,j)=0;
                for (int j=pair_list[i-1][1]; j<pair_list[i][0]; ++j) mask.at<uchar>(x+1,j)=255;
            }
        }
    }

}

/* �ӱ�Եͼ���ϣ�����ƽ���ȡ�ɰ� */
static cv::Mat GetMaskFromEdge(PointCloudConstPtr & cloud, PointCloudNTPtr & normals, cv::Mat & edge)
{
    cv::Mat output = edge.clone();

    /* ��ȡedge������ */
    const cv::Mat contour = ~ edge;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    /* �����ϸ�ж� */
    for (int idx=0; idx<contours.size(); ++idx)
    {
        /* ɨ��ÿһ���� */
        cv::Mat inside(edge.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(inside, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        float nx_min = 2, nx_max = -2,
              ny_min = 2, ny_max = -2,
              nz_min = 2, nz_max = -2,
              x_min = 10000, x_max = -10000,
              y_min = 10000, y_max = -10000,
              z_min = 10000, z_max = -10000;
        /* ע��ij�ķ��� */
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
        /* ͨ���ж��ߴ�, ������ƫת, ȷ��ƽ�� */
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

/* ���ڷ�����ȥ�������е�ƽ�� */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud)
{
    /* ���������� */
    PointCloudNTPtr normals(new pcl::PointCloud<PointNT>);
    normals->width = cloud->width; 
    normals->height = cloud->height;
    normals->resize(normals->width*normals->height);
    /* ���㷨���� */
    pcl::IntegralImageNormalEstimation<PointT, PointNT> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE );
    ne.setDepthDependentSmoothing(false);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    /* ���÷�������Ϣ��ȡ��Ե(��ƽ�沿��) */
    cv::Mat edge = GetEdgeFromDepth(normals);
    /* ��ɫ��������1 */
    OpenImage(edge, 1, 0, 0);
    
    /* �ȳ����޸���С�ĺ�ɫ�� */
    RoughFixMask(cloud, edge);
    /* ���Ժ�ɫ��ƽ������ */
    ConnectDepthMask(cloud, edge);

    /* ͨ���ߴ�ͷ�������ȡ���յ��ɰ� */
    cv::Mat no_plane_mask = GetMaskFromEdge(cloud, normals, edge);
    /* �޸�ConnectDepth; �����ɰ� */
    OpenImage(no_plane_mask, 1, 2, 0);

    return no_plane_mask;
}



}
}

