/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 14.11.2011
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

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include <mrsmap/map/multiresolution_csurfel_map.h>
#include <mrsmap/registration/multiresolution_csurfel_registration.h>

#include <mrsmap/visualization/visualization_map.h>
#include <mrsmap/utilities/utilities.h>

#include <boost/algorithm/string.hpp>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace mrsmap;

// parses Juergen Sturm's datasets (tgz archives + timestamp associations)
// simply takes the base path of the dataset

typedef MultiResolutionColorSurfelMap MultiResolutionSurfelMap;

class EvaluateVisualOdometry {
public:

	EvaluateVisualOdometry( const std::string& path, unsigned int K ) {

		path_ = path;
		K_ = K;

		min_resolution_ = 0.0125f;
		max_radius_ = 30.f;

		alloc_idx_ = 0;

		for( int i = 0; i < 2; i++ ) {
			imageAllocator_[ i ] = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );
			treeNodeAllocator_[ i ] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >(
					new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
		}

	}

	Eigen::Matrix4f processFrame( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud ) {

		pcl::StopWatch stopwatch;

		alloc_idx_ = ( alloc_idx_ + 1 ) % 2;

		// prepare map
		// provide dynamic node allocator
		// use double buffers for image node and tree node allocators
		treeNodeAllocator_[ alloc_idx_ ]->reset();
		boost::shared_ptr< MultiResolutionSurfelMap > currFrameMap = boost::shared_ptr< MultiResolutionSurfelMap >(
				new MultiResolutionSurfelMap( min_resolution_, max_radius_, treeNodeAllocator_[ alloc_idx_ ] ) );

		// add points to local map
		std::vector< int > imageBorderIndices;
		currFrameMap->findVirtualBorderPoints( *cloud, imageBorderIndices );

		currFrameMap->imageAllocator_ = imageAllocator_[ alloc_idx_ ];

		stopwatch.reset();
		currFrameMap->addImage( *cloud );
		currFrameMap->octree_->root_->establishNeighbors();
		currFrameMap->markNoUpdateAtPoints( *cloud, imageBorderIndices );
		currFrameMap->evaluateSurfels();
		double deltat = stopwatch.getTimeSeconds() * 1000.0;
		std::cout << "build: " << deltat << "\n";

		stopwatch.reset();
		currFrameMap->buildShapeTextureFeatures();
		deltat = stopwatch.getTimeSeconds() * 1000.0;
		std::cout << "feature: " << deltat << "\n";

		currFrameMap->findForegroundBorderPoints( *cloud, imageBorderIndices );
		currFrameMap->markBorderAtPoints( *cloud, imageBorderIndices );

		// register frames
		Eigen::Matrix4d transform;
		transform.setIdentity();
//		transform = lastTransform_;

		if( lastFrameMap_ ) {

			stopwatch.reset();
			pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrTgt;
			MultiResolutionColorSurfelRegistration reg;
			reg.estimateTransformation( *lastFrameMap_, *currFrameMap, transform, 32.f * currFrameMap->min_resolution_, currFrameMap->min_resolution_, corrSrc, corrTgt, 100, 0, 5 );

			deltat = stopwatch.getTime();
			std::cout << "register: " << deltat << "\n";

		}

//		transform.setIdentity();

		lastFrameMap_ = currFrameMap;
		lastTransform_ = transform;

		return transform.cast< float >();

	}

	void evaluate() {

		// prepare output file
		// memorize parameters

		// parse associations.txt
		std::ifstream assocFile( ( path_ + std::string( "/associations.txt" ) ).c_str() );

		Eigen::Matrix4f totalTransform;
		totalTransform.setIdentity();

		lastTransform_.setIdentity();

		std::vector< std::vector< std::string > > assocs;

		int count = -1;

		while( assocFile.good() ) {

			count++;

			// read in line
			char lineCStr[ 1024 ];
			assocFile.getline( lineCStr, 1024, '\n' );

			std::string lineStr( lineCStr );

			// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of( "\t " ) );

			if( entryStrs.size() == 4 )
				assocs.push_back( entryStrs );

		}

		lastFrameMap_.reset();
		totalTransform.setIdentity();
		lastTransform_.setIdentity();

		char filenum[ 255 ];
		sprintf( filenum, "%i", K_ );

		std::ofstream outFile( ( path_ + "/" + std::string( "visual_odometry_result_delta" ) + std::string( filenum ) + ".txt" ).c_str() );
		outFile << "# minres: " << min_resolution_ << ", max depth: " << max_radius_ << "\n";

		for( unsigned int t = 1; t < assocs.size(); t += K_ ) {

			std::vector< std::string > entryStrs = assocs[ t ];

			// parse entries, load images, generate point cloud, process images...

			// display last point cloud
			if( lastFrameMap_ ) {
				pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudMap( new pcl::PointCloud< pcl::PointXYZRGBA >() );

				lastFrameMap_->visualize3DColorDistribution( cloudMap, viewer_.selectedDepth, viewer_.selectedViewDir, false );

				pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud2( new pcl::PointCloud< pcl::PointXYZRGBA >() );
				pcl::transformPointCloud( *cloudMap, *cloud2, totalTransform );
				viewer_.displayPointCloud( "map cloud", cloud2 );
			}

			// load images
			cv::Mat depthImg = cv::imread( path_ + "/" + entryStrs[ 1 ], CV_LOAD_IMAGE_ANYDEPTH );
			cv::Mat rgbImg = cv::imread( path_ + "/" + entryStrs[ 3 ], CV_LOAD_IMAGE_ANYCOLOR );

			// extract point cloud from image pair
			pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGBA >() );
			imagesToPointCloud( depthImg, rgbImg, entryStrs[ 0 ], cloud );

			// process data
			pcl::StopWatch stopwatch;
			stopwatch.reset();
			Eigen::Matrix4f transform = processFrame( rgbImg, cloud );
			double processTime = stopwatch.getTimeSeconds() * 1000.0;
			totalTransform = ( totalTransform * transform ).eval();

			std::cout << "total: " << processTime << "\n";

			// write transform to output file
			Eigen::Quaternionf q( Eigen::Matrix3f( totalTransform.block< 3, 3 >( 0, 0 ) ) );
			outFile << entryStrs[ 0 ] << " " << totalTransform( 0, 3 ) << " " << totalTransform( 1, 3 ) << " " << totalTransform( 2, 3 ) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
					<< " " << processTime << "\n";

			lastFrameCloud_ = cloud;

			viewer_.spinOnce();
			usleep( 1000 );

		}

	}

public:

	std::string path_;
	unsigned int K_;

	boost::shared_ptr< MultiResolutionSurfelMap > lastFrameMap_;
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr lastFrameCloud_;

	Eigen::Matrix4d lastTransform_;

	float min_resolution_, max_radius_;

	unsigned int alloc_idx_;
	boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_[ 2 ];
	boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_[ 2 ];

	Viewer viewer_;

};

int main( int argc, char** argv ) {

	po::options_description desc( "Allowed options" );

	std::string inputpath = "";
	int skipframes = 0;

	desc.add_options()( "help,h", "help" )( "inputpath,i", po::value< std::string >( &inputpath )->default_value( "." ), "path to input data" )( "skipframe,s",
			po::value< int >( &skipframes )->default_value( 0 ), "number of skipped frames" );

	po::variables_map vm;
	po::store( po::parse_command_line( argc, argv, desc ), vm );
	po::notify( vm );

	if( vm.count( "help" ) || vm.count( "h" ) ) {
		std::cout << desc << "\n";
		exit( 0 );
	}

	EvaluateVisualOdometry ev( inputpath, skipframes + 1 );
	ev.evaluate();

	while( ev.viewer_.is_running ) {
		ev.viewer_.spinOnce();
		usleep( 1000 );
	}

	return 0;
}

