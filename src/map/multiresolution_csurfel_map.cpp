/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.05.2011
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

#include "mrsmap/map/multiresolution_csurfel_map.h"

#include "octreelib/algorithm/downsample.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include <ostream>
#include <fstream>

using namespace mrsmap;

#define DIST_DEPENDENCY 0.01f

#define MAX_VIEWDIR_DIST cos( 0.25 * M_PI + 0.125*M_PI )

gsl_rng* MultiResolutionColorSurfelMap::r = NULL;
ShapeTextureTable* ShapeTextureTable::instance_ = NULL;

void ShapeTextureTable::initialize() {

	// shape features in [-1,1]
	const float inv_size = 1.f / ( (float) SHAPE_TEXTURE_TABLE_SIZE );
	for( unsigned int i = 0; i < SHAPE_TEXTURE_TABLE_SIZE; i++ ) {

		float s = std::min( ( NUM_SHAPE_BINS - 1.0 ), std::max( 0., ( NUM_SHAPE_BINS - 1.0 ) * ( (float) i ) * inv_size ) );
		const float ds = s - floor( s );

		unsigned int fs = std::max( 0, std::min( NUM_SHAPE_BINS - 1, (int) floor( s ) ) );
		unsigned int cs = std::max( 0, std::min( NUM_SHAPE_BINS - 1, (int) ceil( s ) ) );

		shape_value_table_[ 0 ][ i ].setZero();
		shape_value_table_[ 1 ][ i ].setZero();
		shape_value_table_[ 2 ][ i ].setZero();

		shape_value_table_[ 0 ][ i ]( fs ) = 1.f - ds;
		shape_value_table_[ 0 ][ i ]( cs ) = ds;
		shape_value_table_[ 1 ][ i ]( fs ) = 1.f - ds;
		shape_value_table_[ 1 ][ i ]( cs ) = ds;
		shape_value_table_[ 2 ][ i ]( fs ) = 1.f - ds;
		shape_value_table_[ 2 ][ i ]( cs ) = ds;

		float v = 2.f * ( (float) i ) * inv_size - 1.f;

		float lowl = 0;
		float ctrl = 0;
		float uppl = 0;

		float lowc = 0;
		float ctrc = 0;
		float uppc = 0;

		if( v >= 0 ) {
			if( v >= LUMINANCE_BIN_THRESHOLD )
				uppl = 1.f;
			else {
				uppl = v / LUMINANCE_BIN_THRESHOLD;
				ctrl = 1.f - uppl;
			}

			if( v >= COLOR_BIN_THRESHOLD )
				uppc = 1.f;
			else {
				uppc = v / COLOR_BIN_THRESHOLD;
				ctrc = 1.f - uppc;
			}
		}
		else {

			if( -v >= LUMINANCE_BIN_THRESHOLD )
				lowl = 1.f;
			else {
				lowl = -v / LUMINANCE_BIN_THRESHOLD;
				ctrl = 1.f - lowl;
			}

			if( -v >= COLOR_BIN_THRESHOLD )
				lowc = 1.f;
			else {
				lowc = -v / COLOR_BIN_THRESHOLD;
				ctrc = 1.f - lowc;
			}

		}

		texture_value_table_[ 0 ][ i ].setZero();
		texture_value_table_[ 1 ][ i ].setZero();
		texture_value_table_[ 2 ][ i ].setZero();

		texture_value_table_[ 0 ][ i ]( 0 ) = lowl;
		texture_value_table_[ 0 ][ i ]( 1 ) = ctrl;
		texture_value_table_[ 0 ][ i ]( 2 ) = uppl;

		texture_value_table_[ 1 ][ i ]( 0 ) = lowc;
		texture_value_table_[ 1 ][ i ]( 1 ) = ctrc;
		texture_value_table_[ 1 ][ i ]( 2 ) = uppc;

		texture_value_table_[ 2 ][ i ]( 0 ) = lowc;
		texture_value_table_[ 2 ][ i ]( 1 ) = ctrc;
		texture_value_table_[ 2 ][ i ]( 2 ) = uppc;

	}

}

inline void MultiResolutionColorSurfelMap::ShapeTextureFeature::add( MultiResolutionColorSurfelMap::Surfel* src, MultiResolutionColorSurfelMap::Surfel* dst, float weight ) {

	// surflet pair relation as in "model globally match locally"
	const Eigen::Vector3d p1 = src->mean_.block< 3, 1 >( 0, 0 );
	const Eigen::Vector3d p2 = dst->mean_.block< 3, 1 >( 0, 0 );
	const Eigen::Vector3d n1 = src->normal_;
	const Eigen::Vector3d n2 = dst->normal_;

	Eigen::Vector3d d = p2 - p1;
	d.normalize();

	const int s1 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.5 * ( n1.dot( d ) + 1.0 ) ) ) );
	const int s2 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.5 * ( n2.dot( d ) + 1.0 ) ) ) );
	const int s3 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.5 * ( n1.dot( n2 ) + 1.0 ) ) ) );

	shape_.block< 1, NUM_SHAPE_BINS >( 0, 0 ) += weight * ShapeTextureTable::instance()->shape_value_table_[ 0 ][ s1 ];
	shape_.block< 1, NUM_SHAPE_BINS >( 1, 0 ) += weight * ShapeTextureTable::instance()->shape_value_table_[ 1 ][ s2 ];
	shape_.block< 1, NUM_SHAPE_BINS >( 2, 0 ) += weight * ShapeTextureTable::instance()->shape_value_table_[ 2 ][ s3 ];

	const int c1 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.5 * ( ( dst->mean_( 3, 0 ) - src->mean_( 3, 0 ) ) + 1.0 ) ) ) );
	const int c2 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.25 * ( ( dst->mean_( 4, 0 ) - src->mean_( 4, 0 ) ) + 2.0 ) ) ) );
	const int c3 = std::min( ( SHAPE_TEXTURE_TABLE_SIZE - 1 ), std::max( 0, (int) round( ( SHAPE_TEXTURE_TABLE_SIZE - 1.0 ) * 0.25 * ( ( dst->mean_( 5, 0 ) - src->mean_( 5, 0 ) ) + 2.0 ) ) ) );

	texture_.block< 1, NUM_TEXTURE_BINS >( 0, 0 ) += weight * ShapeTextureTable::instance()->texture_value_table_[ 0 ][ c1 ];
	texture_.block< 1, NUM_TEXTURE_BINS >( 1, 0 ) += weight * ShapeTextureTable::instance()->texture_value_table_[ 1 ][ c2 ];
	texture_.block< 1, NUM_TEXTURE_BINS >( 2, 0 ) += weight * ShapeTextureTable::instance()->texture_value_table_[ 2 ][ c3 ];

	num_points_ += weight;

}

inline void MultiResolutionColorSurfelMap::ShapeTextureFeature::add( const MultiResolutionColorSurfelMap::ShapeTextureFeature& feature, float weight ) {

	shape_ += weight * feature.shape_;
	texture_ += weight * feature.texture_;
	num_points_ += weight * feature.num_points_;

}

MultiResolutionColorSurfelMap::ImagePreAllocator::ImagePreAllocator()
		: imageNodeAllocator_( 10000 ) {
	imgKeys = NULL;
	valueMap = NULL;
	node_image_ = NULL;
	width = height = 0;
}

MultiResolutionColorSurfelMap::ImagePreAllocator::~ImagePreAllocator() {
	if( imgKeys )
		delete[] imgKeys;

	if( valueMap )
		delete[] valueMap;
}

void MultiResolutionColorSurfelMap::ImagePreAllocator::prepare( unsigned int w, unsigned int h, bool buildNodeImage ) {

	typedef NodeValue* NodeValuePtr;
	typedef spatialaggregate::OcTreeNode< float, NodeValue >* NodePtr;

	if( !valueMap || height != h || width != w ) {

		if( imgKeys )
			delete[] imgKeys;
		imgKeys = new uint64_t[ w * h ];

		if( valueMap )
			delete[] valueMap;

		valueMap = new NodeValuePtr[ w * h ];

		if( node_image_ )
			delete[] node_image_;

		if( buildNodeImage )
			node_image_ = new NodePtr[ w * h ];

		infoList.resize( w * h );

		width = w;
		height = h;

	}

	memset( &imgKeys[ 0 ], 0LL, w * h * sizeof(uint64_t) );
	memset( &valueMap[ 0 ], 0, w * h * sizeof(NodeValuePtr) );
	if( buildNodeImage )
		memset( &node_image_[ 0 ], 0, w * h * sizeof(NodePtr) );
	imageNodeAllocator_.reset();

}

void MultiResolutionColorSurfelMap::convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta ) {

	static const float sqrt305 = 0.5f * sqrtf( 3 );

	// RGB to L-alpha-beta:
	// normalize RGB to [0,1]
	// M := max( R, G, B )
	// m := min( R, G, B )
	// L := 0.5 ( M + m )
	// alpha := 0.5 ( 2R - G - B )
	// beta := 0.5 sqrt(3) ( G - B )
	L = 0.5f * ( std::max( std::max( r, g ), b ) + std::min( std::min( r, g ), b ) );
	alpha = 0.5f * ( 2.f * r - g - b );
	beta = sqrt305 * ( g - b );

}

void MultiResolutionColorSurfelMap::convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha * alpha + beta * beta ) ) );
	float s_norm = ( 1.f - fabsf( 2.f * L - 1.f ) );
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		c = s_norm * s;
	}
	else
		c = 0.f;

	if( h < 0 )
		h += 2.f * M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f * floor( 0.5f * h2 );
	float x = c * ( 1.f - fabsf( h_sector - 1.f ) );

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1 + m;
	b = b1 + m;
	g = g1 + m;

}

void convertLAlphaBeta2RGBDamped( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha * alpha + beta * beta ) ) );
	float s_norm = ( 1.f - fabsf( 2.f * L - 1.f ) );
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		// damp saturation stronger when lightness is bad
		s *= expf( -0.5f * 10.f * ( L - 0.5f ) * ( L - 0.5f ) );
		c = s_norm * s;
	}
	else
		c = 0.f;

	if( h < 0 )
		h += 2.f * M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f * floor( 0.5f * h2 );
	float x = c * ( 1.f - fabsf( h_sector - 1.f ) );

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1 + m;
	b = b1 + m;
	g = g1 + m;

}

MultiResolutionColorSurfelMap::MultiResolutionColorSurfelMap( float minResolution, float maxRange, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator ) {

	min_resolution_ = minResolution;
	max_range_ = maxRange;

	Eigen::Matrix< float, 4, 1 > center( 0.f, 0.f, 0.f, 0.f );
	allocator_ = allocator;
	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, NodeValue > >( new spatialaggregate::OcTree< float, NodeValue >( center, minResolution, maxRange, allocator ) );

	if( !r ) {
		const gsl_rng_type* T = gsl_rng_default;
		gsl_rng_env_setup();
		r = gsl_rng_alloc( T );
	}

}

MultiResolutionColorSurfelMap::~MultiResolutionColorSurfelMap() {
}

void MultiResolutionColorSurfelMap::extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov ) {

	std::list< spatialaggregate::OcTreeNode< float, NodeValue >* > nodes;
	octree_->root_->getAllLeaves( nodes );

	Eigen::Matrix< double, 3, 1 > sum;
	Eigen::Matrix< double, 3, 3 > sumSquares;
	double numPoints = 0;
	sum.setZero();
	sumSquares.setZero();

	for( std::list< spatialaggregate::OcTreeNode< float, NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

		NodeValue& v = ( *it )->value_;

		for( int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			Eigen::Vector3d mean_s = v.surfels_[ i ].mean_.block< 3, 1 >( 0, 0 );
			double num_points_s = v.surfels_[ i ].num_points_;

			sum += num_points_s * mean_s;
			sumSquares += num_points_s * ( v.surfels_[ i ].cov_.block< 3, 3 >( 0, 0 ) + mean_s * mean_s.transpose() );
			numPoints += num_points_s;

		}

	}

	if( numPoints > 0 ) {

		const double inv_num = 1.0 / numPoints;
		mean = sum * inv_num;
		cov = inv_num * sumSquares - mean * mean.transpose();

	}

}

void MultiResolutionColorSurfelMap::addPoints( const boost::shared_ptr< const pcl::PointCloud< pcl::PointXYZRGBA > >& cloud, const boost::shared_ptr< const std::vector< int > >& indices ) {
	addPoints( *cloud, *indices );
}

void MultiResolutionColorSurfelMap::addPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f * sqrtf( 3.f );
	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and add point information to map
	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[ indices[ i ] ];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if( isnan( x ) || isinf( x ) )
			continue;

		if( isnan( y ) || isinf( y ) )
			continue;

		if( isnan( z ) || isinf( z ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = *( reinterpret_cast< unsigned int* >( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255 * r, gf = inv_255 * g, bf = inv_255 * b;

		// RGB to L-alpha-beta:
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f * rf - gf - bf );
		float beta = sqrt305 * ( gf - bf );

		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;

		Eigen::Vector3d viewDirection = pos.block< 3, 1 >( 0, 0 ) - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if( viewDistance < 1e-10 )
			continue;

		double viewDistanceInv = 1.0 / viewDistance;
		viewDirection *= viewDistanceInv;

		MultiResolutionColorSurfelMap::Surfel surfel;
		surfel.add( pos );
		surfel.first_view_dir_ = viewDirection;
		surfel.first_view_inv_dist_ = viewDistanceInv;

		NodeValue value;

		// add surfel to view directions within an angular interval
		for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
			const double dist = viewDirection.dot( value.surfels_[ k ].initial_view_dir_ );
			if( dist > max_dist ) {
				value.surfels_[ k ] += surfel;
			}
		}

		// max resolution depends on depth: the farer, the bigger the minimumVolumeSize
		// see: http://www.ros.org/wiki/openni_kinect/kinect_accuracy
		// i roughly used the 90% percentile function for a single kinect
		int depth = ceil( octree_->depthForVolumeSize( std::max( (float) min_resolution_, (float) ( 2.f * DIST_DEPENDENCY * viewDistance * viewDistance ) ) ) );

		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->addPoint( p.getVector4fMap(), value, depth );

	}

}

void MultiResolutionColorSurfelMap::addImage( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, bool smoothViewDir, bool buildNodeImage ) {

	imageAllocator_->prepare( cloud.width, cloud.height, buildNodeImage );
	int imageAggListIdx = 0;

	int idx = 0;
	const unsigned int width4 = 4 * cloud.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[ 0 ];
	NodeValue** mapPtr = &imageAllocator_->valueMap[ 0 ];

	const NodeValue initValue;

	Eigen::Vector4d sensorOrigin = cloud.sensor_origin_.cast< double >();
	const double sox = sensorOrigin( 0 );
	const double soy = sensorOrigin( 1 );
	const double soz = sensorOrigin( 2 );

	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f * sqrtf( 3.f );
	const double max_dist = MAX_VIEWDIR_DIST;

	stopwatch_.reset();

	const float minpx = octree_->min_position_( 0 );
	const float minpy = octree_->min_position_( 1 );
	const float minpz = octree_->min_position_( 2 );

	const float pnx = octree_->position_normalizer_( 0 );
	const float pny = octree_->position_normalizer_( 1 );
	const float pnz = octree_->position_normalizer_( 2 );

	const int maxdepth = octree_->max_depth_;

	const int w = cloud.width;
	const int wm1 = w - 1;
	const int wp1 = w + 1;
	const int h = cloud.height;

	unsigned char depth = maxdepth;
	float minvolsize = octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	for( int y = 0; y < h; y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGBA& p = cloud.points[ idx++ ];

			if( std::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const unsigned int kx_ = ( p.x - minpx ) * pnx + 0.5;
			const unsigned int ky_ = ( p.y - minpy ) * pny + 0.5;
			const unsigned int kz_ = ( p.z - minpz ) * pnz + 0.5;

			const float distdep = ( 2. * DIST_DEPENDENCY * viewDistance * viewDistance );

			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = octree_->depthForVolumeSize( (double) distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = octree_->minVolumeSizeForDepth( depth );
				maxvolsize = octree_->maxVolumeSizeForDepth( depth );

			}

			const unsigned int x_ = ( kx_ >> ( MAX_REPRESENTABLE_DEPTH - depth ) );
			const unsigned int y_ = ( ky_ >> ( MAX_REPRESENTABLE_DEPTH - depth ) );
			const unsigned int z_ = ( kz_ >> ( MAX_REPRESENTABLE_DEPTH - depth ) );

			uint64_t imgkey = ( ( (uint64_t) x_ & 0xFFFFLL ) << 48 ) | ( ( (uint64_t) y_ & 0xFFFFLL ) << 32 ) | ( ( (uint64_t) z_ & 0xFFFFLL ) << 16 ) | (uint64_t) depth;

			// check pixel above
			if( y > 0 ) {

				if( imgkey == *( imgPtr - w ) )
					*mapPtr = *( mapPtr - w );
				else {

					if( imgkey == *( imgPtr - wp1 ) ) {
						*mapPtr = *( mapPtr - wp1 );
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *( imgPtr - wm1 ) ) {
								*mapPtr = *( mapPtr - wm1 );
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *( mapPtr - 1 );
			}

			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.allocate();
				memcpy( ( *mapPtr )->surfels_, initValue.surfels_, sizeof( initValue.surfels_ ) );
				for( unsigned int i = 0; i < 6; i++ ) {
					( *mapPtr )->surfels_[ i ].first_view_dir_ = viewDirection;
					( *mapPtr )->surfels_[ i ].first_view_inv_dist_ = viewDistanceInv;
				}

				ImagePreAllocator::Info& info = imageAllocator_->infoList[ imageAggListIdx ];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				imageAggListIdx++;

			}

			// add point to surfel
			const float rgbf = p.rgb;
			const unsigned int rgb = *( reinterpret_cast< const unsigned int* >( &rgbf ) );
			const unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
			const unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
			const unsigned int b = ( rgb & 0x000000FF );

			// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
			const float rf = inv_255 * (float) r;
			const float gf = inv_255 * (float) g;
			const float bf = inv_255 * (float) b;

			float maxch = rf;
			if( bf > maxch )
				maxch = bf;
			if( gf > maxch )
				maxch = gf;

			float minch = rf;
			if( bf < minch )
				minch = bf;
			if( gf < minch )
				minch = gf;

			const float L = 0.5f * ( maxch + minch );
			const float alpha = 0.5f * ( 2.f * rf - gf - bf );
			const float beta = sqrt305 * ( gf - bf );

			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = L;
			pos( 4 ) = posT( 4 ) = alpha;
			pos( 5 ) = posT( 5 ) = beta;


			if( !smoothViewDir ) {
				MultiResolutionColorSurfelMap::Surfel* surfel = ( *mapPtr )->getSurfel( viewDirection );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
					const double dist = viewDirection.dot( ( *mapPtr )->surfels_[ k ].initial_view_dir_ );
					if( dist > max_dist ) {
						( *mapPtr )->surfels_[ k ].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

	stopwatch_.reset();

	for( unsigned int i = 0; i < imageAggListIdx; i++ ) {

		const ImagePreAllocator::Info& info = imageAllocator_->infoList[ i ];
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
		info.value->association_ = n;

	}

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

	if( buildNodeImage ) {

		imageAllocator_->node_set_.clear();

		NodeValue** mapPtr = &imageAllocator_->valueMap[ 0 ];
		unsigned int idx = 0;

		NodeValue* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[ idx++ ] = ( *mapPtr )->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( ( *mapPtr )->association_ );
					}
				}
				else
					imageAllocator_->node_image_[ idx++ ] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}

void MultiResolutionColorSurfelMap::getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition ) {

	int h = imageAllocator_->height;
	int w = imageAllocator_->width;

	img = cv::Mat( h, w, CV_8UC3, 0.f );

	spatialaggregate::OcTreeNode< float, NodeValue >** nodeImgPtr = &imageAllocator_->node_image_[ 0 ];

	cv::Vec3b v;
	for( int y = 0; y < h; y++ ) {

		for( int x = 0; x < w; x++ ) {

			if( *nodeImgPtr ) {

				float rf = 0, gf = 0, bf = 0;
				Eigen::Vector3d viewDirection = ( *nodeImgPtr )->getPosition().block< 3, 1 >( 0, 0 ).cast< double >() - viewPosition;
				viewDirection.normalize();

				MultiResolutionColorSurfelMap::Surfel* surfel = ( *nodeImgPtr )->value_.getSurfel( viewDirection );
				Eigen::Matrix< double, 6, 1 > vec = ( *nodeImgPtr )->value_.getSurfel( viewDirection )->mean_;

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				v[ 0 ] = b;
				v[ 1 ] = g;
				v[ 2 ] = r;

			}
			else {

				v[ 0 ] = 0;
				v[ 1 ] = 0;
				v[ 2 ] = 0;

			}

			img.at< cv::Vec3b >( y, x ) = v;

			nodeImgPtr++;

		}
	}

}

inline bool MultiResolutionColorSurfelMap::splitCriterion( spatialaggregate::OcTreeNode< float, NodeValue >* oldLeaf, spatialaggregate::OcTreeNode< float, NodeValue >* newLeaf ) {

	return true;

}

void MultiResolutionColorSurfelMap::findImageBorderPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, std::vector< int >& indices ) {

	// determine first image points from the borders that are not nan

	// horizontally
	for( unsigned int y = 0; y < cloud.height; y++ ) {

		for( unsigned int x = 0; x < cloud.width; x++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float x = p.x;
			const float y = p.y;
			const float z = p.z;

			if( isnan( x ) || isinf( x ) )
				continue;

			if( isnan( y ) || isinf( y ) )
				continue;

			if( isnan( z ) || isinf( z ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for( int x = cloud.width - 1; x >= 0; x-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float x = p.x;
			const float y = p.y;
			const float z = p.z;

			if( isnan( x ) || isinf( x ) )
				continue;

			if( isnan( y ) || isinf( y ) )
				continue;

			if( isnan( z ) || isinf( z ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

	// vertically
	for( unsigned int x = 0; x < cloud.width; x++ ) {

		for( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float x = p.x;
			const float y = p.y;
			const float z = p.z;

			if( isnan( x ) || isinf( x ) )
				continue;

			if( isnan( y ) || isinf( y ) )
				continue;

			if( isnan( z ) || isinf( z ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for( int y = cloud.height - 1; y >= 0; y-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float x = p.x;
			const float y = p.y;
			const float z = p.z;

			if( isnan( x ) || isinf( x ) )
				continue;

			if( isnan( y ) || isinf( y ) )
				continue;

			if( isnan( z ) || isinf( z ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

}

void MultiResolutionColorSurfelMap::findVirtualBorderPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, std::vector< int >& indices ) {

	// detect background points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.95f * 0.95f;
	const float invDepthJumpRatio = 1.f / depthJumpRatio;

	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px * px + py * py + pz * pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}

	// vertically
	for( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[ idx ];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px * px + py * py + pz * pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}

}


void MultiResolutionColorSurfelMap::findForegroundBorderPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, std::vector< int >& indices ) {

	// detect foreground points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.95f*0.95f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.clear();
	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGBA& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}

}



void MultiResolutionColorSurfelMap::clearAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[ indices[ i ] ];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if( isnan( x ) || isinf( x ) )
			continue;

		if( isnan( y ) || isinf( y ) )
			continue;

		if( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while( n ) {

			for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[ k ].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[ k ].clear();
				}
			}

			n = n->children_[ n->getOctant( position ) ];
		}

	}

}

void MultiResolutionColorSurfelMap::markNoUpdateAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[ indices[ i ] ];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if( isnan( x ) || isinf( x ) )
			continue;

		if( isnan( y ) || isinf( y ) )
			continue;

		if( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while( n ) {

			for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[ k ].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[ k ].applyUpdate_ = false;
				}
			}

			n = n->children_[ n->getOctant( position ) ];

		}

	}

}

void MultiResolutionColorSurfelMap::clearUpdateSurfelsAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[ indices[ i ] ];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if( isnan( x ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while( n ) {

			for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[ k ].initial_view_dir_ );
				if( dist > max_dist ) {
					if( !n->value_.surfels_[ k ].up_to_date_ ) {
						n->value_.surfels_[ k ].clear();
					}
				}
			}

			n = n->children_[ n->getOctant( position ) ];

		}

	}

}

void MultiResolutionColorSurfelMap::markUpdateAllSurfels() {

	octree_->root_->sweepDown( NULL, &markUpdateAllSurfelsFunction );

}

inline void MultiResolutionColorSurfelMap::markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {

	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ )
		current->value_.surfels_[ i ].applyUpdate_ = true;

}

void MultiResolutionColorSurfelMap::markBorderAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) || isinf( x ) )
			continue;

		if ( isnan( y ) || isinf( y ) )
			continue;

		if ( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while ( n ) {

			n->value_.border_ = true;

			n = n->children_[n->getOctant( position )];

		}

	}

}


inline void MultiResolutionColorSurfelMap::clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	current->value_.border_ = false;

}

void MultiResolutionColorSurfelMap::clearBorderFlag() {

	octree_->root_->sweepDown( NULL, &clearBorderFlagFunction );

}

void MultiResolutionColorSurfelMap::markUpdateImprovedEffViewDistSurfels( const Eigen::Vector3f& viewPosition ) {

	Eigen::Vector3d viewPos = viewPosition.cast< double >();
	octree_->root_->sweepDown( &viewPos, &markUpdateImprovedEffViewDistSurfelsFunction );

}

inline void MultiResolutionColorSurfelMap::markUpdateImprovedEffViewDistSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current,
		spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	const Eigen::Vector3d& viewPos = *( (Eigen::Vector3d*) data );

	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		MultiResolutionColorSurfelMap::Surfel& surfel = current->value_.surfels_[ i ];

		// do we have to switch the flag?
		if( !surfel.applyUpdate_ ) {

			Eigen::Vector3d viewDir = surfel.mean_.block< 3, 1 >( 0, 0 ) - viewPos;
			float effViewDist = viewDir.dot( surfel.initial_view_dir_ ) / viewDir.squaredNorm();
			if( effViewDist > surfel.eff_view_dist_ ) // > since it's inv eff view dist
				surfel.applyUpdate_ = true;

		}

	}

}

void MultiResolutionColorSurfelMap::evaluateSurfels() {

	octree_->root_->sweepDown( NULL, &evaluateSurfelsFunction );

}

void MultiResolutionColorSurfelMap::unevaluateSurfels() {

	octree_->root_->sweepDown( NULL, &unevaluateSurfelsFunction );

}

void MultiResolutionColorSurfelMap::setApplyUpdate( bool v ) {

	octree_->root_->sweepDown( &v, &setApplyUpdateFunction );

}

void MultiResolutionColorSurfelMap::setUpToDate( bool v ) {

	octree_->root_->sweepDown( &v, &setUpToDateFunction );

}

void MultiResolutionColorSurfelMap::clearUnstableSurfels() {

	octree_->root_->sweepDown( NULL, &clearUnstableSurfelsFunction );

}

void MultiResolutionColorSurfelMap::clearAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &clearAssociatedFlagFunction );

}

void MultiResolutionColorSurfelMap::distributeAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &distributeAssociatedFlagFunction );

}

void MultiResolutionColorSurfelMap::clearAssociations() {

	octree_->root_->sweepDown( NULL, &clearAssociationsFunction );

}

inline void MultiResolutionColorSurfelMap::clearAssociationsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
	current->value_.association_ = NULL;
}

bool MultiResolutionColorSurfelMap::pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold ) {

	float queryDepth = position.norm();

	int scale05 = ceil( 0.5f * scale );

	cv::Rect r;
	r.x = (int) floor( imagePoint.x - scale05 );
	r.y = (int) floor( imagePoint.y - scale05 );
	r.width = 2 * scale05;
	r.height = 2 * scale05;

	if( r.x < 0 ) {
		r.width += r.x;
		r.x = 0;
	}

	if( r.y < 0 ) {
		r.height += r.y;
		r.y = 0;
	}

	if( r.x + r.width > image_depth.cols )
		r.width = image_depth.cols - r.x;

	if( r.y + r.height > image_depth.rows )
		r.height = image_depth.rows - r.y;

	cv::Mat patch = image_depth( r );

	// find correponding point for query point in image
	float bestDist = 1e10f;
	int bestX = -1, bestY = -1;
	for( int y = 0; y < patch.rows; y++ ) {
		for( int x = 0; x < patch.cols; x++ ) {
			const float depth = patch.at< float >( y, x );
			if( !isnan( depth ) ) {
				float dist = fabsf( queryDepth - depth );
				if( dist < bestDist ) {
					bestDist = dist;
					bestX = x;
					bestY = y;
				}
			}

		}
	}

	// find depth jumps to the foreground in horizontal, vertical, and diagonal directions
	//	cv::Mat img_show = image_depth.clone();

	for( int dy = -1; dy <= 1; dy++ ) {
		for( int dx = -1; dx <= 1; dx++ ) {

			if( dx == 0 && dy == 0 )
				continue;

			float trackedDepth = queryDepth;
			for( int y = bestY + dy, x = bestX + dx; y >= 0 && y < patch.rows && x >= 0 && x < patch.cols; y += dy, x += dx ) {

				const float depth = patch.at< float >( y, x );
				//				img_show.at<float>(r.y+y,r.x+x) = 0.f;
				if( !isnan( depth ) ) {

					if( trackedDepth - depth > jumpThreshold ) {
						return false;
					}

					trackedDepth = depth;

				}

			}

		}
	}

	return true;
}

void MultiResolutionColorSurfelMap::buildShapeTextureFeatures() {

	octree_->root_->sweepDown( NULL, &buildSimpleShapeTextureFeatureFunction );
	octree_->root_->sweepDown( NULL, &buildAgglomeratedShapeTextureFeatureFunction );

}

inline void MultiResolutionColorSurfelMap::buildSimpleShapeTextureFeatureFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {

	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		current->value_.surfels_[ i ].simple_shape_texture_features_.initialize();

		if( current->value_.surfels_[ i ].num_points_ < MIN_SURFEL_POINTS )
			continue;

		current->value_.surfels_[ i ].simple_shape_texture_features_.add( &current->value_.surfels_[ i ], &current->value_.surfels_[ i ], current->value_.surfels_[ i ].num_points_ );

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[ n ] ) {

				if( current->neighbors_[ n ]->value_.surfels_[ i ].num_points_ <= MIN_SURFEL_POINTS )
					continue;

				current->value_.surfels_[ i ].simple_shape_texture_features_.add( &current->value_.surfels_[ i ], &current->neighbors_[ n ]->value_.surfels_[ i ],
						current->neighbors_[ n ]->value_.surfels_[ i ].num_points_ );

			}

		}

	}

}

inline void MultiResolutionColorSurfelMap::buildAgglomeratedShapeTextureFeatureFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current,
		spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	const float neighborFactor = 0.1f;

	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		current->value_.surfels_[ i ].agglomerated_shape_texture_features_ = current->value_.surfels_[ i ].simple_shape_texture_features_;

		if( current->value_.surfels_[ i ].num_points_ < MIN_SURFEL_POINTS )
			continue;

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[ n ] ) {

				if( current->neighbors_[ n ]->value_.surfels_[ i ].num_points_ < MIN_SURFEL_POINTS ) {
					continue;
				}

				current->value_.surfels_[ i ].agglomerated_shape_texture_features_.add( current->neighbors_[ n ]->value_.surfels_[ i ].simple_shape_texture_features_,
						current->neighbors_[ n ]->value_.surfels_[ i ].simple_shape_texture_features_.num_points_ * neighborFactor );

			}

		}

		if( current->value_.surfels_[ i ].agglomerated_shape_texture_features_.num_points_ > 0.5f ) {
			float inv_num = 1.f / current->value_.surfels_[ i ].agglomerated_shape_texture_features_.num_points_;
			current->value_.surfels_[ i ].agglomerated_shape_texture_features_.shape_ *= inv_num;
			current->value_.surfels_[ i ].agglomerated_shape_texture_features_.texture_ *= inv_num;
		}

		current->value_.surfels_[ i ].agglomerated_shape_texture_features_.num_points_ = 1.f;

	}

}

void MultiResolutionColorSurfelMap::clearAssociationDist() {
	octree_->root_->sweepDown( NULL, &clearAssociationDistFunction );
}

inline void MultiResolutionColorSurfelMap::clearAssociationDistFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		current->value_.surfels_[ i ].assocDist_ = std::numeric_limits< float >::max();
	}
}

inline void MultiResolutionColorSurfelMap::setApplyUpdateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	bool v = *( (bool*) data );
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		if( current->value_.surfels_[ i ].num_points_ >= MIN_SURFEL_POINTS ) {
			current->value_.surfels_[ i ].applyUpdate_ = v;
		}
	}
}

inline void MultiResolutionColorSurfelMap::setUpToDateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	bool v = *( (bool*) data );
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		current->value_.surfels_[ i ].up_to_date_ = v;
	}
}

inline void MultiResolutionColorSurfelMap::clearUnstableSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		if( current->value_.surfels_[ i ].num_points_ < MIN_SURFEL_POINTS ) {
			// reinitialize
			current->value_.surfels_[ i ].up_to_date_ = false;
			current->value_.surfels_[ i ].mean_.setZero();
			current->value_.surfels_[ i ].cov_.setZero();
			current->value_.surfels_[ i ].num_points_ = 0;
			current->value_.surfels_[ i ].applyUpdate_ = true;
		}
	}
}

inline void MultiResolutionColorSurfelMap::evaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	current->value_.evaluateSurfels();
}

inline void MultiResolutionColorSurfelMap::unevaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	current->value_.unevaluateSurfels();
}

inline void MultiResolutionColorSurfelMap::clearAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
}

inline void MultiResolutionColorSurfelMap::distributeAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {

	for( unsigned int n = 0; n < 27; n++ ) {

		if( current->neighbors_[ n ] && current->neighbors_[ n ]->value_.associated_ == 0 ) {
			current->neighbors_[ n ]->value_.associated_ = 2;
		}

	}

}

std::vector< unsigned int > MultiResolutionColorSurfelMap::findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, int maxDepth ) {

	std::vector< unsigned int > inliers;
	inliers.reserve( indices.size() );

	const float max_mahal_dist = 12.59f;

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f * sqrtf( 3.f );

	Eigen::Vector3d sensorOrigin;
	for( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	// project each point into map and find inliers
	// go through the point cloud and add point information to map
	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGBA& p = cloud.points[ indices[ i ] ];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if( isnan( x ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = *( reinterpret_cast< unsigned int* >( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255 * r, gf = inv_255 * g, bf = inv_255 * b;

		// RGB to L-alpha-beta:
		// normalize RGB to [0,1]
		// M := max( R, G, B )
		// m := min( R, G, B )
		// L := 0.5 ( M + m )
		// alpha := 0.5 ( 2R - G - B )
		// beta := 0.5 sqrt(3) ( G - B )
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f * rf - gf - bf );
		float beta = sqrt305 * ( gf - bf );

		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;

		Eigen::Vector3d viewDirection = pos.block< 3, 1 >( 0, 0 ) - sensorOrigin;
		viewDirection.normalize();

		Eigen::Vector4f pos4f = pos.block< 4, 1 >( 0, 0 ).cast< float >();

		// lookup node for point
		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = octree_->root_->findRepresentative( pos4f, maxDepth );

		MultiResolutionColorSurfelMap::Surfel* surfel = n->value_.getSurfel( viewDirection );
		if( surfel->num_points_ > MIN_SURFEL_POINTS ) {

			// inlier? check mahalanobis distance
			Eigen::Matrix< double, 6, 6 > invcov = surfel->cov_.inverse();
			Eigen::Matrix< double, 6, 1 > diff = surfel->mean_.block< 6, 1 >( 0, 0 ) - pos;

			if( diff.dot( invcov * diff ) < max_mahal_dist ) {
				inliers.push_back( i );
			}

		}

	}

}

struct Visualize3DColorDistributionInfo {
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr;
	int viewDir, depth;
	bool random;
};

void MultiResolutionColorSurfelMap::visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr, int depth, int viewDir, bool random ) {

	Visualize3DColorDistributionInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;

	octree_->root_->sweepDown( &info, &visualize3DColorDistributionFunction );

}

inline void MultiResolutionColorSurfelMap::visualize3DColorDistributionFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {

	Visualize3DColorDistributionInfo* info = (Visualize3DColorDistributionInfo*) data;

	if( ( info->depth == -1 && current->type_ != spatialaggregate::OCTREE_LEAF_NODE ) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

//	std::cout << current->resolution() << "\n";

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();

	// generate markers for histogram surfels
	for( unsigned int i = 0; i < 6; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap::Surfel& surfel = current->value_.surfels_[ i ];

		if( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		if( info->random ) {

			// samples N points from the normal distribution in mean and cov...
			unsigned int N = 100;

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();

			for( unsigned int j = 0; j < N; j++ ) {

				Eigen::Matrix< double, 6, 1 > vec;
				for( unsigned int k = 0; k < 6; k++ )
					vec( k ) = gsl_ran_gaussian( r, 1.0 );

				vec( 3 ) = vec( 4 ) = vec( 5 ) = 0.0;

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGBA p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = *( reinterpret_cast< float* >( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}

		}
		else {

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();

			std::vector< Eigen::Matrix< double, 6, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 1 > > > vecs;

			Eigen::Matrix< double, 6, 1 > v;
			v.setZero();

			vecs.push_back( v );

			v( 0 ) = 1.0;
			v( 1 ) = 0.0;
			v( 2 ) = 0.0;
			vecs.push_back( v );

			v( 0 ) = -1.0;
			v( 1 ) = 0.0;
			v( 2 ) = 0.0;
			vecs.push_back( v );

			v( 0 ) = 0.0;
			v( 1 ) = 1.0;
			v( 2 ) = 0.0;
			vecs.push_back( v );

			v( 0 ) = 0.0;
			v( 1 ) = -1.0;
			v( 2 ) = 0.0;
			vecs.push_back( v );

			v( 0 ) = 0.0;
			v( 1 ) = 0.0;
			v( 2 ) = 1.0;
			vecs.push_back( v );

			v( 0 ) = 0.0;
			v( 1 ) = 0.0;
			v( 2 ) = -1.0;
			vecs.push_back( v );

			v( 0 ) = 1.0;
			v( 1 ) = 1.0;
			v( 2 ) = 1.0;
			vecs.push_back( v );

			v( 0 ) = 1.0;
			v( 1 ) = 1.0;
			v( 2 ) = -1.0;
			vecs.push_back( v );

			v( 0 ) = 1.0;
			v( 1 ) = -1.0;
			v( 2 ) = 1.0;
			vecs.push_back( v );

			v( 0 ) = 1.0;
			v( 1 ) = -1.0;
			v( 2 ) = -1.0;
			vecs.push_back( v );

			v( 0 ) = -1.0;
			v( 1 ) = 1.0;
			v( 2 ) = 1.0;
			vecs.push_back( v );

			v( 0 ) = -1.0;
			v( 1 ) = 1.0;
			v( 2 ) = -1.0;
			vecs.push_back( v );

			v( 0 ) = -1.0;
			v( 1 ) = -1.0;
			v( 2 ) = 1.0;
			vecs.push_back( v );

			v( 0 ) = -1.0;
			v( 1 ) = -1.0;
			v( 2 ) = -1.0;
			vecs.push_back( v );

			for( unsigned int k = 0; k < vecs.size(); k++ ) {

				Eigen::Matrix< double, 6, 1 > vec = 1.1 * vecs[ k ];

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGBA p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = *( reinterpret_cast< float* >( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}
		}
	}

}

struct VisualizePrincipalSurfaceInfo {
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr;
	int viewDir, depth;
	MultiResolutionColorSurfelMap* map;
	std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > samples;
};

void MultiResolutionColorSurfelMap::visualizePrincipalSurface( pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr, int depth, int viewDir ) {

	VisualizePrincipalSurfaceInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.map = this;

	// 2N+1 sample points in each dimension
	int N = 5;
	for( int dx = -N; dx <= N; dx++ ) {
		for( int dy = -N; dy <= N; dy++ ) {
			info.samples.push_back( Eigen::Vector3d( (float) dx / (float) N * 0.5f, (float) dy / (float) N * 0.5f, 0 ) );
		}
	}

	octree_->root_->sweepDown( &info, &visualizePrincipalSurfaceFunction );

}

inline void MultiResolutionColorSurfelMap::visualizePrincipalSurfaceFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next,
		void* data ) {

	VisualizePrincipalSurfaceInfo* info = (VisualizePrincipalSurfaceInfo*) data;

	if( current->depth_ != info->depth )
		return;

	float resolution = current->resolution();

	for( unsigned int i = 0; i < 6; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap::Surfel& surfel = current->value_.surfels_[ i ];

		if( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		// project samples to principal surface of GMM in local neighborhood using subspace constrained mean shift
		std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* > neighbors;
		neighbors.reserve( 27 );
		current->getNeighbors( neighbors );

		std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > centerPositions;
		std::vector< MultiResolutionColorSurfelMap::Surfel* > surfels;
		surfels.reserve( neighbors.size() );
		for( unsigned int j = 0; j < neighbors.size(); j++ ) {

			if( !neighbors[ j ] )
				continue;

			if( neighbors[ j ]->value_.surfels_[ i ].num_points_ < MIN_SURFEL_POINTS )
				continue;
			else {

				// precalculate centerpos of neighbor node
				Eigen::Vector3d centerPosN = neighbors[ j ]->getCenterPosition().block< 3, 1 >( 0, 0 ).cast< double >();

				surfels.push_back( &neighbors[ j ]->value_.surfels_[ i ] );
				centerPositions.push_back( centerPosN );
			}
		}

		Eigen::Vector3d centerPos = current->getCenterPosition().block< 3, 1 >( 0, 0 ).cast< double >();

		Eigen::Vector3d meani = surfel.mean_.block< 3, 1 >( 0, 0 );
		Eigen::Matrix3d covi = surfel.cov_.block< 3, 3 >( 0, 0 );

		if( covi.determinant() <= std::numeric_limits< double >::epsilon() )
			continue;

		// eigen decompose covariance to find rotation onto principal plane
		// eigen vectors are stored in the columns in ascending order
		Eigen::Matrix3d eigenVectors;
		Eigen::Vector3d eigenValues;
		pcl::eigen33( covi, eigenVectors, eigenValues );

		Eigen::Matrix3d R_cov;
		R_cov.setZero();
		R_cov.block< 3, 1 >( 0, 0 ) = eigenVectors.col( 2 );
		R_cov.block< 3, 1 >( 0, 1 ) = eigenVectors.col( 1 );
		R_cov.block< 3, 1 >( 0, 2 ) = eigenVectors.col( 0 );

		// include resolution scale
		R_cov *= 1.2f * resolution;

		for( unsigned int j = 0; j < info->samples.size(); j++ ) {
			Eigen::Vector3d sample = meani + R_cov * info->samples[ j ];

			if( info->map->projectOnPrincipalSurface( sample, surfels, centerPositions, resolution ) ) {

				// dont draw in other node volumes
				if( ( centerPos - sample ).maxCoeff() < 0.55f * resolution && ( centerPos - sample ).minCoeff() > -0.55f * resolution ) {

					// conditional mean color in GMM

					Eigen::Vector3d meanSum;
					meanSum.setZero();
					double weightSum = 0;

					for( unsigned int k = 0; k < surfels.size(); k++ ) {

						Eigen::Vector3d means = surfels[ k ]->mean_.block< 3, 1 >( 0, 0 );
						Eigen::Matrix3d covs = surfels[ k ]->cov_.block< 3, 3 >( 0, 0 );
						Eigen::Matrix3d covs_raw = covs;
						covs *= INTERPOLATION_COV_FACTOR;

						if( covs.determinant() <= std::numeric_limits< double >::epsilon() )
							continue;

						Eigen::Vector3d centerDiff = centerPositions[ k ] - sample;
						const double dx = resolution - fabsf( centerDiff( 0 ) );
						const double dy = resolution - fabsf( centerDiff( 1 ) );
						const double dz = resolution - fabsf( centerDiff( 2 ) );
						if( dx < 0 || dy < 0 || dz < 0 )
							continue;

						double weight = dx * dy * dz;

						Eigen::Matrix3d invcovs = covs.inverse();

						Eigen::Vector3d us = invcovs * ( sample - means );
						double dist = exp( -0.5 * ( sample - means ).dot( us ) );

						double prob = 1.0 / sqrt( 8.0 * M_PI * M_PI * M_PI * covs.determinant() ) * dist;

						Eigen::Vector3d meanc = surfels[ k ]->mean_.block< 3, 1 >( 3, 0 );
						const Eigen::Matrix3d cov_cs = surfels[ k ]->cov_.block< 3, 3 >( 3, 0 );
						const Eigen::Vector3d mean_cond_cs = meanc + cov_cs * covs_raw.inverse() * ( sample - means );

						meanSum += weight * prob * mean_cond_cs;
						weightSum += weight * prob;
					}

					if( weightSum > 0 ) {

						meanSum /= weightSum;

						pcl::PointXYZRGBA p;
						p.x = sample( 0 );
						p.y = sample( 1 );
						p.z = sample( 2 );

						const float L = meanSum( 0 );
						const float alpha = meanSum( 1 );
						const float beta = meanSum( 2 );

						float rf = 0, gf = 0, bf = 0;
						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

						int rgb = ( r << 16 ) + ( g << 8 ) + b;
						p.rgb = *( reinterpret_cast< float* >( &rgb ) );

						info->cloudPtr->points.push_back( p );

					}

				}

			}

		}

	}

}

bool MultiResolutionColorSurfelMap::projectOnPrincipalSurface( Eigen::Vector3d& sample, const std::vector< MultiResolutionColorSurfelMap::Surfel* >& neighbors,
		const std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& centerPositions, double resolution ) {

	if( neighbors.size() == 0 )
		return false;

	Eigen::Vector3d x = sample;

	int maxIterations = 10;
	double epsilon = 1e-4;

	Eigen::Matrix3d covadd;
	covadd.setIdentity();

	int it = 0;
	while( it < maxIterations ) {

		// evaluate pdf and gradients at each mixture component
		double prob = 0;
		Eigen::Vector3d grad;
		grad.setZero();
		Eigen::Matrix3d hess;
		hess.setZero();
		Eigen::Matrix3d covSum;
		covSum.setZero();
		Eigen::Vector3d meanSum;
		meanSum.setZero();
		double weightSum = 0;

		double max_dist = 0.0;

		// use trilinear interpolation weights

		for( unsigned int i = 0; i < neighbors.size(); i++ ) {

			Eigen::Vector3d meani = neighbors[ i ]->mean_.block< 3, 1 >( 0, 0 );
			Eigen::Matrix3d covi = neighbors[ i ]->cov_.block< 3, 3 >( 0, 0 );

			covi *= INTERPOLATION_COV_FACTOR;

			if( covi.determinant() <= std::numeric_limits< double >::epsilon() )
				continue;

			Eigen::Vector3d centerDiff = centerPositions[ i ] - x;
			const double dx = resolution - fabsf( centerDiff( 0 ) );
			const double dy = resolution - fabsf( centerDiff( 1 ) );
			const double dz = resolution - fabsf( centerDiff( 2 ) );
			if( dx < 0 || dy < 0 || dz < 0 )
				continue;

			double weight = dx * dy * dz;

			Eigen::Matrix3d invcovi = covi.inverse();

			Eigen::Vector3d ui = invcovi * ( x - meani );
			double dist = exp( -0.5 * ( x - meani ).dot( ui ) );
			double probi = 1.0 / sqrt( 8.0 * M_PI * M_PI * M_PI * covi.determinant() ) * dist;
			max_dist = std::max( max_dist, dist );

			prob += weight * probi;
			grad -= weight * probi * ui;
			hess += weight * probi * ( ui * ( ui.transpose() ).eval() - invcovi );

			meanSum += weight * probi * invcovi * meani;
			covSum += weight * probi * invcovi;

			weightSum += weight;
		}

		if( isnan( weightSum ) ) {
			return false;
		}

		prob /= weightSum;
		grad /= weightSum;
		hess /= weightSum;

		if( prob > 1e-12 && ( it < maxIterations - 1 || max_dist > 0.05 / INTERPOLATION_COV_FACTOR ) ) {

			Eigen::Vector3d mean = covSum.inverse() * meanSum;
			Eigen::Matrix3d invcov = -1.0 / prob * hess + 1.0 / ( prob * prob ) * grad * ( grad.transpose() ).eval();

			// eigen decomposition of invcov
			// eigen vectors are stored in the columns in ascending order
			Eigen::Matrix3d eigenVectors;
			Eigen::Vector3d eigenValues;
			pcl::eigen33( invcov, eigenVectors, eigenValues );

			Eigen::Matrix< double, 3, 3 > V_ortho;
			V_ortho.setZero();
			V_ortho.block< 3, 1 >( 0, 0 ) = eigenVectors.col( 2 );

			x = ( x + V_ortho * ( V_ortho.transpose() ).eval() * ( mean - x ).eval() ).eval();

			// stopping criterion
			if( fabsf( grad.dot( V_ortho.transpose() * grad ) ) / ( grad.norm() * ( V_ortho.transpose() * grad ).norm() ) < epsilon )
				break;

		}
		else
			return false;

		it++;

	}

	sample = x;
	return true;

}

// s. http://people.cs.vt.edu/~kafura/cs2704/op.overloading2.html
template< typename T, int rows, int cols >
std::ostream& operator<<( std::ostream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for( unsigned int i = 0; i < rows; i++ ) {
		for( unsigned int j = 0; j < cols; j++ ) {
			T d = m( i, j );
			os.write( (char*) &d, sizeof(T) );
		}
	}

	return os;
}

template< typename T, int rows, int cols >
std::istream& operator>>( std::istream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for( unsigned int i = 0; i < rows; i++ ) {
		for( unsigned int j = 0; j < cols; j++ ) {
			T d;
			os.read( (char*) &d, sizeof(T) );
			m( i, j ) = d;
		}
	}

	return os;
}

std::ostream& operator<<( std::ostream& os, MultiResolutionColorSurfelMap::NodeValue& v ) {

	for( int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		os << v.surfels_[ i ].initial_view_dir_;
		os << v.surfels_[ i ].first_view_dir_;
		os.write( (char*) &v.surfels_[ i ].first_view_inv_dist_, sizeof(float) );
		os.write( (char*) &v.surfels_[ i ].num_points_, sizeof(double) );
		os << v.surfels_[ i ].mean_;
		os << v.surfels_[ i ].normal_;
		os << v.surfels_[ i ].cov_;
		os.write( (char*) &v.surfels_[ i ].up_to_date_, sizeof(bool) );
		os.write( (char*) &v.surfels_[ i ].applyUpdate_, sizeof(bool) );
		os.write( (char*) &v.surfels_[ i ].idx_, sizeof(int) );

	}

	return os;
}

std::istream& operator>>( std::istream& os, MultiResolutionColorSurfelMap::NodeValue& v ) {

	for( int i = 0; i < MAX_NUM_SURFELS; i++ ) {
		os >> v.surfels_[ i ].initial_view_dir_;
		os >> v.surfels_[ i ].first_view_dir_;
		os.read( (char*) &v.surfels_[ i ].first_view_inv_dist_, sizeof(float) );
		os.read( (char*) &v.surfels_[ i ].num_points_, sizeof(double) );
		os >> v.surfels_[ i ].mean_;
		os >> v.surfels_[ i ].normal_;
		os >> v.surfels_[ i ].cov_;
		os.read( (char*) &v.surfels_[ i ].up_to_date_, sizeof(bool) );
		os.read( (char*) &v.surfels_[ i ].applyUpdate_, sizeof(bool) );
		os.read( (char*) &v.surfels_[ i ].idx_, sizeof(int) );

	}

	return os;
}

std::ostream& operator<<( std::ostream& os, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >& node ) {

	os.write( (char*) &node.depth_, sizeof(int) );
	os.write( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );
	os << node.value_;

	return os;
}

std::istream& operator>>( std::istream& os, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >& node ) {

	os.read( (char*) &node.depth_, sizeof(int) );
	os.read( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );
	os >> node.value_;

	return os;

}

void MultiResolutionColorSurfelMap::save( const std::string& filename ) {

	// create downsampling map for the target
	algorithm::OcTreeSamplingMap< float, MultiResolutionColorSurfelMap::NodeValue > samplingMap = algorithm::downsampleOcTree( *octree_, false, octree_->max_depth_ );

	std::ofstream outfile( filename.c_str(), std::ios::out | std::ios::binary );

	// header information
	outfile.write( (char*) &min_resolution_, sizeof(float) );
	outfile.write( (char*) &max_range_, sizeof(float) );

	for( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodes = samplingMap[ i ].size();
		outfile.write( (char*) &numNodes, sizeof(int) );

		for( std::list< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = samplingMap[ i ].begin(); it != samplingMap[ i ].end(); ++it ) {
			outfile << *( *it );
		}
	}

}

void MultiResolutionColorSurfelMap::load( const std::string& filename ) {

	std::ifstream infile( filename.c_str(), std::ios::in | std::ios::binary );

	if( !infile.is_open() ) {
		std::cout << "could not open file " << filename.c_str() << "\n";
	}

	infile.read( (char*) &min_resolution_, sizeof(float) );
	infile.read( (char*) &max_range_, sizeof(float) );

	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, NodeValue > >(
			new spatialaggregate::OcTree< float, NodeValue >( Eigen::Matrix< float, 4, 1 >( 0.f, 0.f, 0.f, 0.f ), min_resolution_, max_range_ ) );
	octree_->allocator_->deallocateNode( octree_->root_ );
	octree_->root_ = NULL;

	for( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodesOnDepth = 0;
		infile.read( (char*) &numNodesOnDepth, sizeof(int) );

		for( int j = 0; j < numNodesOnDepth; j++ ) {

			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node = octree_->allocator_->allocateNode();
			octree_->acquire( node );

			infile >> ( *node );

			// insert octree node into the tree
			// start at root and traverse the tree until we find an empty leaf
			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = octree_->root_;

			if( !n ) {
				node->parent_ = NULL;
				octree_->root_ = node;
			}
			else {

				// search for parent
				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n2 = n;
				while( n2 ) {
					n = n2;
					n2 = n->children_[ n->getOctant( node->pos_key_ ) ];
				}

				// assert that found parent node has the correct depth
				if( n->depth_ != node->depth_ - 1 || n->type_ != spatialaggregate::OCTREE_BRANCHING_NODE ) {
					std::cout << "MultiResolutionMap::load(): bad things happen\n";
				}
				else {
					n->children_[ n->getOctant( node->pos_key_ ) ] = node;
					node->parent_ = n;
				}
			}

		}

	}

}

