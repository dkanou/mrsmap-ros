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

#include "mrsmap/utilities/utilities.h"

using namespace mrsmap;

bool mrsmap::pointInImage( const Eigen::Vector4f& p ) {

	if( isnan( p(0) ) )
		return false;

	double px = 525.0 * p(0) / p(2);
	double py = 525.0 * p(1) / p(2);


	if( px < -320.0 || px > 320.0 || py < -240.0 || py > 240.0 ) {
		return false;
	}

	return true;

}

void mrsmap::imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud, unsigned int downsampling ) {

	cloud->header.frame_id = "openni_rgb_optical_frame";
	cloud->is_dense = true;
	cloud->height = depthImg.rows / downsampling;
	cloud->width = depthImg.cols / downsampling;
	cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
	cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
	cloud->points.resize( cloud->height * cloud->width );

	const float invfocalLength = 1.f / 525.f;
	const float centerX = 319.5f;
	const float centerY = 239.5f;
	const float factor = 1.f / 5000.f;

	const unsigned short* depthdata = reinterpret_cast< const unsigned short* >( &depthImg.data[ 0 ] );
	const unsigned char* colordata = &colorImg.data[ 0 ];
	int idx = 0;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {
		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( x % downsampling != 0 || y % downsampling != 0 ) {
				colordata += 3;
				depthdata++;
				continue;
			}

			pcl::PointXYZRGBA& p = cloud->points[ idx ];

			if( *depthdata == 0 || factor * (float) ( *depthdata ) > 10.f ) {
				p.x = std::numeric_limits< float >::quiet_NaN();
				p.y = std::numeric_limits< float >::quiet_NaN();
				p.z = std::numeric_limits< float >::quiet_NaN();
			}
			else {
				float xf = x;
				float yf = y;
				float dist = factor * (float) ( *depthdata );
				p.x = ( xf - centerX ) * dist * invfocalLength;
				p.y = ( yf - centerY ) * dist * invfocalLength;
				p.z = dist;
			}

			depthdata++;

			int b = ( *colordata++ );
			int g = ( *colordata++ );
			int r = ( *colordata++ );

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = *( reinterpret_cast< float* >( &rgb ) );

			idx++;

		}
	}

}

void mrsmap::getCameraCalibration( cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs ) {

	distortionCoeffs = cv::Mat( 1, 5, CV_32FC1, 0.f );
	cameraMatrix = cv::Mat( 3, 3, CV_32FC1, 0.f );

	cameraMatrix.at< float >( 0, 0 ) = 525.f;
	cameraMatrix.at< float >( 1, 1 ) = 525.f;
	cameraMatrix.at< float >( 2, 2 ) = 1.f;

	cameraMatrix.at< float >( 0, 2 ) = 319.5f;
	cameraMatrix.at< float >( 1, 2 ) = 239.5f;

}

void mrsmap::pointCloudToImage( const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& cloud, cv::Mat& img ) {

	img = cv::Mat( cloud->height, cloud->width, CV_8UC3, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGBA& p = cloud->points[ idx ];

			cv::Vec3b px;
			px[ 0 ] = p.b;
			px[ 1 ] = p.g;
			px[ 2 ] = p.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}

void mrsmap::pointCloudsToOverlayImage( const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& rgb_cloud, const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& overlay_cloud, cv::Mat& img ) {

	img = cv::Mat( rgb_cloud->height, rgb_cloud->width, CV_8UC3, 0.f );

	float alpha = 0.2;

	int idx = 0;
	for( unsigned int y = 0; y < rgb_cloud->height; y++ ) {
		for( unsigned int x = 0; x < rgb_cloud->width; x++ ) {

			const pcl::PointXYZRGBA& p1 = rgb_cloud->points[ idx ];
			const pcl::PointXYZRGBA& p2 = overlay_cloud->points[ idx ];

			cv::Vec3b px;
			px[ 0 ] = ( 1 - alpha ) * p1.b + alpha * p2.b;
			px[ 1 ] = ( 1 - alpha ) * p1.g + alpha * p2.g;
			px[ 2 ] = ( 1 - alpha ) * p1.r + alpha * p2.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}

