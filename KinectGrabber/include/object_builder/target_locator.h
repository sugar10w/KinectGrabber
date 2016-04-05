/*
 * Created by sugar10w, 2016.1.19
 * Last edited by sugar10w, 2016.2.25
 *
 * ������תչ̨����״��ȷ��ת�������λ�á�
 * ���������λ��ȡ���ֲ�ͼ���򻯼�������
 * �����תչ̨��ת�٣�����ֲ����Ƶ���ת��
 *
 * TODO Kinect2��5cm�߶����壬��ͬƽ�渽��û�����������������ʱ���ֲ������Ϣ����2���Ļ���
 *
 */

#ifndef _OBJECTBUILDER_LOCATECENTER_
#define _OBJECTBUILDER_LOCATECENTER_

#include"common.h"

namespace tinker {
namespace vision {   

const float RAW_Z_ZOOM =  0.5f;  //TODO �ߴ����10cm�����壬ʹ��1;  ���С��10cm�����壬ʹ��0.5
const float RAW_RADIUS_Z = 4.0f; //TODO ��תչ̨�뾶����ҪZ_ZOOM����
const float ROTATING_SPEED =  6.28 / 62.5; // ��תչ̨˳ʱ����ת

const float CUT_WIDTH = 25.0f,   // �ü���ˮƽֱ��
            CUT_HEIGHT = 30.0f,  // �ü��ø߶�
            EXTRA_DEPTH = 1.0f;  // �ü��ã�������ת������

void ResetZZoom(float zoom = 1.0f);

PointT LocateCenterPoint(PointCloudPtr cloud);

PointCloudPtr CutNearCenter(PointCloudPtr cloud, PointT center);

PointCloudPtr MoveToCenter(PointCloudPtr cloud, PointT center);

PointCloudPtr RotateAfterTime(PointCloudPtr cloud, float time);

}
}

#endif //_OBJECTBUILDER_LOCATECENTER_
