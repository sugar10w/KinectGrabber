/*
 * Created by sugar10w, 2016.3.19
 * Last edited by sugar10w, 2016.5.1 
 *
 * �ӽṹ�����ƣ���ά�������еĵ��ƣ���
 * �����ƽ���ɰ�
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
 * TODO Kinect�ɼ���ͼ��ı�Ե���Ļ���ǳ���, �������仯���������ᳬ�������趨����ֵ, ���±�Ե���ֵ�ƽ�治������ʶ��! */
static cv::Mat GetEdgeFromDepth(PointCloudNTPtr & cloud)
{
    cv::Mat edge(cloud->height, cloud->width, CV_8UC1);
    
    /* ע��ͳһ����ĸ�ʽ */
    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j)
        {
            /* ����ϵĵ㲻����, ��ֹԽ�� */
            if (j==0 || j+1==cloud->width 
                || i==0 || i+1==cloud->height)
            {
                edge.at<uchar>(i, j)=0;
                ++k;
                continue;
            }

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

            ++k;
        }
    return edge;
}

/* �����̫С�ĺ�ɫ���������,
 * ((����ȶ)���ҵ���ȥ�����ƽ��) */
static void RoughFixMask(PointCloudConstPtr &cloud, cv::Mat & mask)
{
    const cv::Mat contour = ~mask;
    
    //cv::Mat fill;
    //cv::cvtColor(mask, fill, CV_GRAY2RGB);

    //��ȡ����
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    //int max_plane_idx = -1;
    /*double max_plane_area = 10000;*/

    //�����С��ֱ��������
    for (int idx=0; idx<contours.size(); ++idx)
    {
        if (contours[idx].size()>1800) continue;

        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size<300) 
        {
            cv::drawContours(mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
            continue;
        }
        //if (cont_size > max_plane_area)
        //{
        //    //max_plane_area = cont_size;
        //    //max_plane_idx = idx;

        //    cv::Mat fill_C1(mask.size(), CV_8UC1, cv::Scalar(0));
        //    cv::drawContours(fill_C3, contours, idx, cv::Scalar(255), CV_FILLED, 4);

        //    //cv::imshow("max_plane", fill_C1);

        //    PointCloudPtr max_plane = GetCloudFromMask(fill_C1, cloud);
        //    PlaneFilter plane_filter(max_plane, 0.02);
        //
        //    if (plane_filter.PlaneFound())
        //    {
        //        cv::Mat plane_mask = plane_filter.GetMask(cloud, 0.05);
        //        mask &= plane_mask;
        //    }

        //    std::cout<<"Contour "<<cont_size<<" has just been removed."<<std::endl;
        //}
    }
    /*std::cout<<std::endl;*/

/*    if (max_plane_idx != -1)
    *///{

    //}

    //cv::cvtColor(fill, mask, CV_RGB2GRAY);
}

/* Ϊ�˴���ǰ��ϸС����ĸ��Ŷ����������зֵ���������Զ����ڵ�������б����ʽ�����Ӳ����� */
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
static float Distance(pcl::PointXYZ unit_vector, pcl::PointXYZ vector)
{
    pcl::PointXYZ result;
    result.x = unit_vector.y * vector.z - unit_vector.z * vector.y;
    result.y = unit_vector.z * vector.x - unit_vector.x * vector.z;
    result.z = unit_vector.x * vector.y - unit_vector.y * vector.x;
    return sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
}
static void ConnectDepthMask(PointCloudConstPtr & cloud, cv::Mat & mask)
{
    /* ˮƽ�����ȡһ���ֵ��߽��д���ע���ʱ��ɫ����Ч�㡣 */
    const int box_size = 5;
    const float dist_limit = 0.02f;

    for (int x=box_size; x<cloud->height-1; x+=box_size)
    {
        //�ȼ����ȡ������������
        std::vector<cv::Vec2i> pair_list;

        for (int y=0; y<cloud->width; ++y)
        {
            if (mask.at<uchar>(x,y)) continue;
            
            //ֱ�ӻ�ȡ�����߶η�Χ
            int start = y, end = y+1;
            while ( end<cloud->width && !mask.at<uchar>(x, end)) ++end;
            --end;

            //����߶κϷ��ԡ���Ч�߶�������(0,0)��
            if (end-start<=5) { pair_list.push_back(cv::Vec2i(0,0)); y=end+1; continue;}
            bool valid = true;
            const PointT & start_point = cloud->points[(x+1)*cloud->width-start-1];
            pcl::PointXYZ unit_vector = GetVector(start_point,cloud->points[(x+1)*cloud->width-end-1]);
            for (int i=start+1; i<end; ++i)
                if (Distance(unit_vector, 
                    GetVector(start_point, cloud->points[(x+1)*cloud->width-i-1], false))
                        > dist_limit)
                {
                    valid = false;
                    break;
                }
            if (valid) pair_list.push_back(cv::Vec2i(start, end));
                 else  pair_list.push_back(cv::Vec2i(0,0));
            y = end;
        }

        //���������߶β�����
        for (int i=1; i<pair_list.size(); ++i)
        {
            if (pair_list[i][0]==0 && pair_list[i][1]==0 ) continue;
            if (pair_list[i-1][0]==0 && pair_list[i-1][1]==0 ) continue;

            bool bind = true;
            PointT start_point = cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1];
            pcl::PointXYZ unit_vector = GetVector(
                cloud->points[(x+1)*cloud->width-pair_list[i-1][0]-1],
                cloud->points[(x+1)*cloud->width-pair_list[i-1][1]-1]);
            for (int j=pair_list[i][0]; j<pair_list[i][1]; ++j)
                if (Distance(unit_vector,
                    GetVector(start_point, cloud->points[(x+1)*cloud->width-j-1], false))
                        > dist_limit)
                {
                    bind = false;
                    break;
                }
            if (bind)
            {
                //���������ߣ���ֹ����
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
    const cv::Mat contour = ~ edge;

    cv::Mat output = edge.clone();

    //��ȡ����
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    //�����ϸ�ж�
    for (int idx=0; idx<contours.size(); ++idx)
    {
        cv::Mat inside(edge.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(inside, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        float nx_min = 2, nx_max = -2,
              ny_min = 2, ny_max = -2,
              nz_min = 2, nz_max = -2,
              x_min = 10000, x_max = -10000,
              y_min = 10000, y_max = -10000,
              z_min = 10000, z_max = -10000;
        // ע��ͳһ����ĸ�ʽ
        int k = 0;
        for (int i=0; i<cloud->height; ++i)
            for (int j=cloud->width-1; j>=0; --j)
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
                ++k;
            }
        if (! ( (x_max-x_min)>0.6
            || (y_max-y_min)>0.6
            || (z_max-z_min)>0.6
            || ( (nx_max-nx_min)<0.01 && (ny_max-ny_min)<0.01 && (nz_max-nx_min)<0.01 ))) //TODO
        {
            cv::drawContours(output, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        }
        //else
        //{
        //    OpenImage(inside, 4, 0, 0);
        //    fill &= ~inside;
        //}
    }

    return  output;
}

/* ���ڷ�����ȥ�������е�ƽ�� */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud)
{
    /* ���㷨���� */
    PointCloudNTPtr normals(new pcl::PointCloud<PointNT>);
    normals->width = cloud->width; 
    normals->height = cloud->height;
    normals->resize(normals->width*normals->height);

    pcl::IntegralImageNormalEstimation<PointT, PointNT> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE );
    ne.setDepthDependentSmoothing(false);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    /* ���÷�������Ϣ��ȡ��Ե(��ƽ�沿��) */
    cv::Mat edge = GetEdgeFromDepth(normals);
    /* ��ɫ��������1 */
    OpenImage(edge, 1, 0, 0);
    
    /* �ȳ����޸� */
    RoughFixMask(cloud, edge);
    /* ����ƽ������ */
    ConnectDepthMask(cloud, edge);

    /* �ӱ�Եͼ�����ҵ����Ǳ�����Сƽ��飬��Ϊ��ƽ���ɰ� */
    cv::Mat no_plane_mask = GetMaskFromEdge(cloud, normals, edge);
    OpenImage(no_plane_mask, 1, 0, 0);

    return no_plane_mask;
}



}
}

