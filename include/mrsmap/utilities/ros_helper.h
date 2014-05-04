/*
 * ros_helper.h
 *
 *  Created on: May 3, 2014
 *      Author: core
 */

#ifndef ROS_HELPER_H_
#define ROS_HELPER_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ros/conversions.h>
#include "pcl_ros/point_cloud.h" //Important
#include "pcl/visualization/pcl_visualizer.h"
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>

namespace mrsmap {

class ROSHelper {
public:
	void updateByPointCloud(const string& idx, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud);
	void advertisePointCloud2();
};

}

#endif /* ROS_HELPER_H_ */
