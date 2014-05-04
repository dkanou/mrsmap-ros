/*
 * ros_helper.cpp
 *
 *  Created on: May 3, 2014
 *      Author: core
 */

#include "mrsmap/utilities/ros_helper.h"

#include <vector>
#include <map>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace mrsmap {

class ROSHelper {
private:
	std::map<string, int> index_;
	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> clouds_;
	ros::NodeHandle nh_;
	ros::Publisher pub_;
};

ROSHelper::ROSHelper() {
	pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/mrsmap/slam", 1);
}

void ROSHelper::updateByPointCloud(const string& idx, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud) {
	std::map<string, int>::iterator it = index_.find(idx);
	if(it != index_.end()) {
		clouds_[index_[idx]] = cloud;
	} else {
		index_[idx] = clouds_.size();
		clouds_.push_back(cloud);
	}
	return true;
}

void ROSHelper::advertisePointCloud2() {
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = (new pcl::PointCloud<pcl::PointXYZRGBA>());
	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> it;
	for(it = clouds_.begin(); it != clouds_.end(); it++) {
		*cloud = *cloud + **it;
	}
	cloud->header.frame_id = "kinect_base";
	pub_.publish(*cloud);
}




}
