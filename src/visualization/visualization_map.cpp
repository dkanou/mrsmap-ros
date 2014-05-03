/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 24.09.2012
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "mrsmap/visualization/visualization_map.h"

#include "pcl/common/common_headers.h"

using namespace mrsmap;

Viewer::Viewer(ros::NodeHandle* n) {

	viewer = boost::shared_ptr< pcl::visualization::PCLVisualizer >( new pcl::visualization::PCLVisualizer( "MRS Viewer" ) );
	viewer->setBackgroundColor( 1, 1, 1 );
	viewer->addCoordinateSystem( 0.2 );
	viewer->initCameraParameters();

	viewer->registerKeyboardCallback( &Viewer::keyboardEventOccurred, *this, NULL );

	selectedDepth = 10; // d
	selectedViewDir = -1; // v
	processFrame = true; // p
	displayScene = true; // s
	displayMap = true; // m
	displayCorr = false; // c
	displayAll = true; // a
	recordFrame = false; // r
	forceRedraw = false; // f

	is_running = true;

	if(n != NULL)
		nh = n;
	else {
		nh = new ros::NodeHandle();
	}
	pub = nh->advertise<sensor_msgs::PointCloud2>("/mrsmap", 1);
}

Viewer::~Viewer() {
}

void Viewer::spinOnce() {

	if( !viewer->wasStopped() ) {
		viewer->spinOnce( 1 );
	}
	else {
		is_running = false;
	}

}

void Viewer::displayPointCloud( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud, int pointSize ) {

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	pcl::copyPointCloud( *cloud, *cloud2 );

	for( unsigned int i = 0; i < cloud2->points.size(); i++ )
		if( isnan( cloud2->points[ i ].x ) ) {
			cloud2->points[ i ].x = 0;
			cloud2->points[ i ].y = 0;
			cloud2->points[ i ].z = 0;
		}

	pcl::visualization::PointCloudColorHandlerRGBField< pcl::PointXYZRGBA > rgb = pcl::visualization::PointCloudColorHandlerRGBField< pcl::PointXYZRGBA >( cloud2 );

	if( !viewer->updatePointCloud< pcl::PointXYZRGBA >( cloud2, rgb, name ) ) {
		viewer->addPointCloud< pcl::PointXYZRGBA >( cloud2, rgb, name );
	}
	viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, name );
}



void Viewer::displayPose( const Eigen::Matrix4d& pose ) {

	static int poseidx = 0;

	double axislength = 0.2;

	pcl::PointXYZRGBA p1, p2;

	char str[ 255 ];

	if( poseidx > 0 ) {
		sprintf( str, "posex%i", poseidx - 1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posex%i", poseidx );
	p1.x = pose( 0, 3 );
	p1.y = pose( 1, 3 );
	p1.z = pose( 2, 3 );
	p2.x = p1.x + axislength * pose( 0, 0 );
	p2.y = p1.y + axislength * pose( 1, 0 );
	p2.z = p1.z + axislength * pose( 2, 0 );
	viewer->addLine( p1, p2, 1.0, 0.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );

	if( poseidx > 0 ) {
		sprintf( str, "posey%i", poseidx - 1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posey%i", poseidx );
	p1.x = pose( 0, 3 );
	p1.y = pose( 1, 3 );
	p1.z = pose( 2, 3 );
	p2.x = p1.x + axislength * pose( 0, 1 );
	p2.y = p1.y + axislength * pose( 1, 1 );
	p2.z = p1.z + axislength * pose( 2, 1 );
	viewer->addLine( p1, p2, 0.0, 1.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );

	if( poseidx > 0 ) {
		sprintf( str, "posez%i", poseidx - 1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posez%i", poseidx );
	p1.x = pose( 0, 3 );
	p1.y = pose( 1, 3 );
	p1.z = pose( 2, 3 );
	p2.x = p1.x + axislength * pose( 0, 2 );
	p2.y = p1.y + axislength * pose( 1, 2 );
	p2.z = p1.z + axislength * pose( 2, 2 );
	viewer->addLine( p1, p2, 0.0, 0.0, 1.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );

	poseidx++;

}

void Viewer::displayCorrespondences( const std::string& name, const pcl::PointCloud< pcl::PointXYZ >::Ptr& cloud1, const pcl::PointCloud< pcl::PointXYZ >::Ptr& cloud2 ) {

	std::vector< int > indices( cloud1->points.size() );
	for( unsigned int i = 0; i < indices.size(); i++ )
		indices[ i ] = i;

	viewer->removeCorrespondences( name );
	viewer->addCorrespondences< pcl::PointXYZ >( cloud1, cloud2, indices, name );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, name );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5.0, 0.0, 0.0, name );

}

void Viewer::keyboardEventOccurred( const pcl::visualization::KeyboardEvent &event, void* data ) {

	if( ( event.getKeySym() == "d" || event.getKeySym() == "D" ) && event.keyDown() ) {

		if( event.getKeySym() == "d" ) {
			selectedDepth++;
		}
		else {
			selectedDepth--;
			if( selectedDepth < 0 )
				selectedDepth = 15;
		}

		selectedDepth = selectedDepth % 16;
		std::cout << "Selected Depth " << selectedDepth << "\n";
	}
	if( ( event.getKeySym() == "v" || event.getKeySym() == "V" ) && event.keyDown() ) {

		if( event.getKeySym() == "v" ) {
			selectedViewDir++;
			if( selectedViewDir == 7 )
				selectedViewDir = -1;
		}
		else {
			selectedViewDir--;
			if( selectedViewDir < -1 )
				selectedViewDir = 6;
		}

		std::cout << "Selected View Dir " << selectedViewDir << "\n";

	}
	if( ( event.getKeySym() == "p" ) && event.keyDown() ) {
		processFrame = true;
	}
	if( ( event.getKeySym() == "s" ) && event.keyDown() ) {
		displayScene = !displayScene;
	}
	if( ( event.getKeySym() == "m" ) && event.keyDown() ) {
		displayMap = !displayMap;
	}
	if( ( event.getKeySym() == "a" ) && event.keyDown() ) {
		displayAll = !displayAll;
	}
	if( ( event.getKeySym() == "c" ) && event.keyDown() ) {
		displayCorr = !displayCorr;
	}
	if( ( event.getKeySym() == "f" ) && event.keyDown() ) {
		forceRedraw = true;
	}
}

void Viewer::advertisePointCloud2(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) {
	sensor_msgs::PointCloud2 pc2;
	pcl::toROSMsg(*cloud, pc2);
	pc2.header.frame_id = "kinect_base";
	pub.publish(pc2);
}
