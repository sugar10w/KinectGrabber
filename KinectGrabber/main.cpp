/*
 * Created by UnaNancyOwen, 2015.11 (https://github.com/UnaNancyOwen/KinectGrabber/)
 * Last edited by sugar10w, 2016.5.7
 * 
 * tinker2016的子项目. 
 * (1) 显示典型三维场景, 尤其是桌面或者书架场景中, 找到的物体.
 * (2) 连接蓝牙姿态传感器, 重建房间的三维模型.
 *
 */

// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "kinect2_grabber.h"
#include "windows.h"

#include <boost/make_shared.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

#include "pointcloud_filter/2dmask_filter.h"
#include "pointcloud_filter/rgbd_plane_detector.h"
#include "pointcloud_recognition/2d_cluster_divider.h"
#include "pointcloud_recognition/edge_cluster_divider.h"
#include "common.h"

using namespace tinker::vision;
using namespace std;

/* 点云的显示 viewer */
pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer( "3D Object Detector" ) );

/* 同viewer关联的键盘控制开关 */
bool flag_all = false;      /* a 显示全部的点云    */
bool flag_paused = false;   /* p 暂停接收点云数据  */
bool flag_draw_box = false; /* d 绘制BonudingBox   */
bool flag_debug = false;    /* b 显示关键蒙版      */
bool flag_color = false;    /* v 显示2d结果        */
bool flag_integral = false; /* i 积分模式          */
bool flag_lock = false;     /* l 锁定蒙版          */
bool flag_viewpoint = false;/* z 接收蓝牙姿态信息  */
bool flag_sigma = false;    /* x 在有姿态信息的情况下进行积分 */

/* 处理过程中的各个关键蒙版 */
cv::Mat mask(424, 512, CV_8UC1, cv::Scalar(0));             /* 过滤结果 */
cv::Mat shard_mask(424, 512, CV_8UC1, cv::Scalar(0));       /* 三维边缘 */ 
cv::Mat no_plane_mask(424, 512, CV_8UC1, cv::Scalar(0));    /* 去平面   */
const cv::Mat mask_all(424, 512, CV_8UC1, cv::Scalar(255)); /* 不过滤   */
cv::Mat mask_downsample(424, 512, CV_8UC1, cv::Scalar(0));  /* 降采样用 */
/* BGR图像 */
cv::Mat img(424, 512, CV_8UC3, cv::Scalar(0,0,0));          /* 原始彩色图像 */
cv::Mat img_result(424, 512, CV_8UC3, cv::Scalar(0,0,0));   /* 彩色图像的处理结果 */

/* 目标的序列 */
vector<ObjectCluster> divided_objects;

/* 积分模式设定 */
cv::Mat integral_mask(424, 512, CV_8UC1, cv::Scalar(0));
int count_frames = 0;
const int MAX_COUNT_FRAMES = 80;

/* 在有姿态信息的情况下, 用于累计的点云 */
PointCloudPtr cloud_sigma(new PointCloud);
/* 姿态变换矩阵 */
Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();

/* 通过内存共享, 获取蓝牙姿态信息 */
LPCWSTR strMapName = L"BluetoothAngleSharedMemory";
HANDLE hMap = NULL;
LPVOID pBuffer = NULL;
bool MemShareIntialize()
{
    hMap = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, strMapName);
    if (NULL == hMap)
    {
        cout<< "警告, 没有接收到姿态信号!" <<endl;
        return false;
    }
    cout << "姿态信号已连接. " <<endl;
    return true;
}
void MemShareGet()
{
    if (NULL == hMap)
    {
        flag_viewpoint = false;
        cout << "姿态信息意外掉线!(hMap) 请检查蓝牙与串口模块. " << endl;
        return;
    }
    pBuffer = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (NULL == pBuffer)
    {
        hMap = NULL;
        flag_viewpoint = false;
        cout << "姿态信息意外掉线!(pBuffer) 请检查蓝牙与串口模块. " << endl;
        return;
    }
    
    double* d = (double*)pBuffer;
    double alpha, beta, gamma;
    /*PointCloudPtr new_cloud(new PointCloud); */
    alpha = d[0]*3.1415926/180;
    beta  = d[1]*3.1415926/180;
    gamma = d[2]*3.1415926/180;
    
    transform_matrix << 
        cos(beta)*cos(gamma)  , -cos(alpha)*sin(beta)*cos(gamma)-sin(alpha)*sin(gamma) , -sin(alpha)*sin(beta)*cos(gamma)+cos(alpha)*sin(gamma)  , 0 ,
        sin(beta)             , cos(beta)*cos(beta)                                    , sin(alpha)*cos(beta)                                  , 0 ,
        -cos(beta)*sin(gamma) , cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma)  , sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma) , 0 ,
        0                     , 0                                                      , 0                                                      , 1 ;

    /*pcl::transformPointCloud(*cloud, *new_cloud, transform_matrix);
    cloud = new_cloud;*/

    //cout << "读取共享内存数据：" << d[0]<<"\t"<<d[1]<<"\t"<<d[2] << endl;

}
void RotateCloud(PointCloudPtr& cloud)
{
    PointCloudPtr new_cloud(new PointCloud);
    pcl::transformPointCloud(*cloud, *new_cloud, transform_matrix);
    cloud = new_cloud;
}

/* viewer的键盘响应 */
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
    if (event.getKeySym() == "a" && event.keyDown())
    {
        flag_all = !flag_all;
    }
    if (event.getKeySym() == "p" && event.keyDown())
    {
        flag_paused = !flag_paused;

        if (flag_paused)
        {
            flag_lock = true;
            cout<<"已停止接收点云信息. 已锁定结果蒙版. 已暂停对蒙版的计算."<<endl;
        }
        else
        {
            flag_lock = false;
            cout<<"开始接收点云信息. 已取消对结果蒙版的锁定."<<endl;
        }
    }
    if (event.getKeySym() == "d" && event.keyDown())
    {
        flag_draw_box = !flag_draw_box;
        if (flag_draw_box)
        {
            viewer->removeAllShapes();
            for (int i=0; i<divided_objects.size(); ++i)
                divided_objects[i].DrawBoundingBox(*viewer, i);
            cout<<"找到"<<divided_objects.size()<<"个目标."<<endl;
        }
        else
        {
            viewer->removeAllShapes();
        }
    }
    if (event.getKeySym() == "b" && event.keyDown())
    {
        flag_debug = !flag_debug;
        if (flag_debug)
        {
            cv::imshow("no_plane_mask", no_plane_mask);
            cv::imshow("shard_mask", shard_mask);
        }
    }
    if (event.getKeySym() == "v" && event.keyDown())
    {
        flag_color = !flag_color;
        if (flag_color)
        {
            cv::imshow("raw RGB", img);
            cv::imshow("2D Result", img_result);
        }
    }
    if (event.getKeySym() == "i" && event.keyDown())
    {
        if (flag_paused || flag_lock)
        {
            cout << "请先解除p/l锁定, 再进入i模式. "<<endl;
        }
        else
        {
            flag_integral = !flag_integral;

            if (flag_integral)
            {
                integral_mask = 0;
                cout<<"已进入积分模式, 开始采集"<<MAX_COUNT_FRAMES<<"帧图像."<<endl;
            }
            else
            {
                count_frames = 0;
                cout<<"已退出积分模式. "<<endl;
            }
        }
    }
    if (event.getKeySym() == "l" && event.keyDown())
    {
        if (flag_paused)
        {
            cout<<"请先退出p状态(暂停接收点云), 再尝试锁定结果蒙版! "<<endl;
        }
        else
        {
            flag_lock = !flag_lock;
        
            if (flag_lock)
                cout<<"已锁定结果蒙版. 已暂停对蒙版的计算."<<endl;
            else
                cout<<"已取消对结果蒙版的锁定. "<<endl;
        }
    }

    if (event.getKeySym() == "z" && event.keyDown())
    {
        if (flag_viewpoint)
        {
            flag_viewpoint = false;
            cout<< "已停止姿态追踪." <<endl;
        }
        else
        {
            if (MemShareIntialize())
            {
                flag_viewpoint = true;
                cout << "已开始姿态追踪." <<endl;
            }
            else
            {
                flag_viewpoint = false;
                cout << "q操作失败, 请检查蓝牙与串口模块. " <<endl;
            }
        }
    }
    if (event.getKeySym() == "x" && event.keyDown())
    {
        if (flag_viewpoint)
        {
            if (flag_sigma)
            {
                flag_sigma = false;
                pcl::io::savePCDFileBinary("sigma.pcd", *cloud_sigma);
                
                PointCloudPtr null_cloud(new PointCloud);
                cloud_sigma = null_cloud;

                cout<< "场景采集结束, 已保存到\"sigma.pcd\"." <<endl;
            }
            else
            {   
                flag_sigma = true;
                cout<< "开始抽样全方位场景." <<endl;
            }
        }
        else
        {
            flag_sigma = false;
            cout<< "请先开启姿态跟踪(z), 再尝试全方向积分(x)." << endl;
        }
    }

    if (event.getKeySym() == "r" && event.keyDown())
    {
        viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
        cout<<"视角已重置."<<endl;
    }
}

int main( int argc, char* argv[] )
{
    /* 设置viewer基本参数 */
    viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
    viewer->setBackgroundColor(0.8, 0.8, 0.8);
    /* 关联键盘响应 */
    viewer->registerKeyboardCallback(keyboardEventOccurred);

    /* ConstPtr点云 */
    pcl::PointCloud<PointT>::ConstPtr cloud;

    /* 设置进程锁(MUTual EXclusion); 设置从Grabber获取点云的函数; */
    boost::mutex mutex;
    boost::function<void( const pcl::PointCloud<PointT>::ConstPtr& )> function =
        [&cloud, &mutex]( const pcl::PointCloud<PointT>::ConstPtr& ptr ){
            boost::mutex::scoped_lock lock( mutex );
            cloud = ptr;
        };

    /* 设置Kinect2Grabber; 关联获取点云函数; 开始连接Kinect2; */
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();
    boost::signals2::connection connection = grabber->registerCallback( function );
    grabber->start();

    /* mask_downsample用于flag_sigma(x)的降采样 */
    for (int i=0; i<424; i+=3)
        for (int j=0; j<512; j+=3)
            mask_downsample.at<uchar>(i,j)=255;

    /* 主循环 */
    while( !viewer->wasStopped() )
    {
        /* 更新viewer */
        viewer->spinOnce();

        /* 若flag_paused, 不处理点云 */
        if (flag_paused)
        {
            viewer->spinOnce(200);
            continue;
        }

        /* 检查进程锁 */
        boost::mutex::scoped_try_lock lock( mutex );
        if( lock.owns_lock() && cloud && cloud->size() != 0 )
        {
            /* 在z模式下, 获取姿态信息 */
            if (flag_viewpoint)
            {
                MemShareGet();
            }

            /* 在mask没有被锁定的情况下, 计算mask */
            if (!flag_lock)
            {
                /* 准备处理点云 */
                divided_objects.clear();

                /* 去平面 */
                no_plane_mask = GetNoPlaneMask(cloud);
                if (flag_debug) cv::imshow("no_plane_mask", no_plane_mask);

                /* 三维边缘 */
                shard_mask = ~GetShardMask(cloud);
                if (flag_debug) cv::imshow("shard_mask", shard_mask);

                /* 方案1: EdgeClusterDivider */
                EdgeClusterDivider cluster_divider_0(cloud, shard_mask, no_plane_mask);
                cluster_divider_0.GetDividedCluster(divided_objects);
                /* 更新蒙版 */
                mask = cluster_divider_0.GetMask();

                /* 积分模式(i) */
                if (flag_integral)
                {
                    /* 蒙版的累积 */
                    for (int i=0; i<integral_mask.rows; ++i)
                        for (int j=0; j<integral_mask.cols; ++j)
                        {
                            int t = integral_mask.at<uchar>(i,j);
                            t = t * count_frames;
                            if (mask.at<uchar>(i,j)) t+=255;
                            t /= (count_frames+1);
                            if (t>255) t=255;
                            integral_mask.at<uchar>(i,j) = t;

                            mask.at<uchar>(i,j) = (t>128)? 255:0;
                        }
                    /* 记录帧数 */
                    ++count_frames;
                    cout<<count_frames<<"\t";
                    if (count_frames == MAX_COUNT_FRAMES)
                    {
                        flag_lock = true;
                        flag_integral = false;
                        count_frames = 0;
                        cout<<endl<<"积分模式已结束. 积分蒙版已初始化. 结果蒙版已锁定(已进入l状态). "<<endl<<endl;
                    }
                    cv::imshow("integral",integral_mask);

                    /* 重新获取目标序列 */
                    ClusterDivider2D cd(cloud, mask);
                    divided_objects.clear();
                    cd.GetDividedCluster(divided_objects);
                }
            }

            /* 显示点云 */
            PointCloudPtr display_cloud;
            if (flag_all) display_cloud = GetCloudFromMask(mask_all, cloud);
                        else display_cloud = GetCloudFromMask(mask, cloud);
            if (flag_viewpoint)
            {
                RotateCloud(display_cloud);
                if (flag_sigma)
                {
                    /* 降采样之后再累加 */
                    PointCloudPtr cloud_downsample;
                    if (flag_all) cloud_downsample = GetCloudFromMask(mask_downsample, cloud);
                             else cloud_downsample = GetCloudFromMask(mask_downsample & mask, cloud);
                    RotateCloud(cloud_downsample);
                    *cloud_sigma += *cloud_downsample;
                    *display_cloud += *cloud_sigma;
                }
            }
            if( !viewer->updatePointCloud( display_cloud, "cloud" ) )
                viewer->addPointCloud(display_cloud, "cloud");

            if (flag_color)
            {
                /* 获取二维BGR图片 img */
                const PointT* pt = & cloud->points[0];
                for (int i=0; i<cloud->height; ++i)
                    for (int j=cloud->width-1; j>=0; --j)
                    {
                        cv::Vec3b & img_point = img.at<cv::Vec3b>(i, j);
                        img_point[0] = pt->b;
                        img_point[1] = pt->g;
                        img_point[2] = pt->r;
                        ++pt;
                    }
                cv::imshow("raw RGB", img);

                /* 处理二维过滤结果 img */
                cv::Mat exp_mask = mask.clone();
                OpenImage(exp_mask, 3, 0, 0);

                img_result = img.clone();
                for (int i=0; i<mask.rows; ++i)
                    for (int j=0; j<mask.cols; ++j)
                        if (! mask.at<uchar>(i,j) && !exp_mask.at<uchar>(i,j))
                        {
                            cv::Vec3b & point = img_result.at<cv::Vec3b>(i,j);
                            /* 背景部分 灰度, 变暗 */
                            point[0] = point[1] = point[2] = 
                                (point[0]+point[1]+point[2])/6;
                        }
                        else if (!mask.at<uchar>(i,j) &&  exp_mask.at<uchar>(i,j))
                        {
                            cv::Vec3b & point = img_result.at<cv::Vec3b>(i,j);
                            /* 绘制白色边框 */
                            point[0] = point[1] = point[2] = 255;
                        }
                cv::imshow("2D Result", img_result);
            }

            if (flag_draw_box && !flag_lock)
            {
                viewer->removeAllShapes();
                for (int i=0; i<divided_objects.size(); ++i)
                    divided_objects[i].DrawBoundingBox(*viewer, i);
                cout<<"找到"<<divided_objects.size()<<"个目标."<<endl;
            }       
        }
        else
        {
            if (!flag_lock)
            {
                SYSTEMTIME sys_time;   
                GetLocalTime( &sys_time ); 
                cout<<" 警告: 点云接收失败 ["<< sys_time.wSecond << "."<< sys_time.wMilliseconds <<"]"<<endl;
            }
        }
       
    }

    /* 结束 */
    grabber->stop();

    return 0;
}
