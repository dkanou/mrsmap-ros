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

#ifndef MULTIRESOLUTION_CSURFEL_MAP_H_
#define MULTIRESOLUTION_CSURFEL_MAP_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>

#include <octreelib/spatialaggregate/octree.h>

#include <gsl/gsl_rng.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/time.h>

#define MAX_NUM_SURFELS 6

#define MIN_SURFEL_POINTS 10.0
#define MAX_SURFEL_POINTS 10000.0

#define NUM_SHAPE_BINS 3
#define NUM_TEXTURE_BINS 3

#define LUMINANCE_BIN_THRESHOLD 0.1
#define COLOR_BIN_THRESHOLD 0.05

#define SHAPE_TEXTURE_TABLE_SIZE 10000

#define INTERPOLATION_COV_FACTOR 20.0

namespace mrsmap {

class MultiResolutionColorSurfelMap {
public:

	class Surfel;

	class ShapeTextureFeature {
	public:

		ShapeTextureFeature() {
			initialize();
		}

		~ShapeTextureFeature() {
		}

		void initialize() {
			shape_.setZero();
			texture_.setZero();
			num_points_ = 0.f;
		}

		inline void add( Surfel* src, Surfel* dst, float weight );
		inline void add( const ShapeTextureFeature& feature, float weight );

		inline float textureDistance( const ShapeTextureFeature& feature ) const {
			return ( texture_ - feature.texture_ ).squaredNorm();
		}

		inline float shapeDistance( const ShapeTextureFeature& feature ) const {
			return ( shape_ - feature.shape_ ).squaredNorm();
		}

		inline float distance( const ShapeTextureFeature& feature ) const {
			return ( shape_ - feature.shape_ ).squaredNorm() + ( texture_ - feature.texture_ ).squaredNorm();
		}

		EIGEN_ALIGN16 Eigen::Matrix< float, 3, NUM_SHAPE_BINS > shape_;
		EIGEN_ALIGN16 Eigen::Matrix< float, 3, NUM_TEXTURE_BINS > texture_;
		float num_points_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

	class Surfel {
	public:
		Surfel() {
			clear();
		}

		~Surfel() {
		}

		inline void clear() {

			num_points_ = 0.0;
			mean_.setZero();
			cov_.setZero();

			up_to_date_ = false;
			applyUpdate_ = true;
			unevaluated_ = false;

			eff_view_dist_ = std::numeric_limits< float >::max();

			assocWeight_ = 1.f;

			idx_ = -1;

		}

		inline Surfel& operator+=( const Surfel& rhs ) {

			if( rhs.num_points_ > 0 && num_points_ < MAX_SURFEL_POINTS ) {

				// numerically stable one-pass update scheme
				if( num_points_ == 0 ) {
					cov_ = rhs.cov_;
					mean_ = rhs.mean_;
					num_points_ = rhs.num_points_;
				}
				else {
					const Eigen::Matrix< double, 6, 1 > deltaS = rhs.num_points_ * mean_ - num_points_ * rhs.mean_;
					cov_ += rhs.cov_ + 1.0 / (num_points_ * rhs.num_points_ * (rhs.num_points_ + num_points_)) * deltaS * deltaS.transpose();
					mean_ += rhs.mean_;
					num_points_ += rhs.num_points_;
				}

				first_view_dir_ = rhs.first_view_dir_;
				first_view_inv_dist_ = rhs.first_view_inv_dist_;
				up_to_date_ = false;
			}

			return *this;
		}

		inline void add( const Eigen::Matrix< double, 6, 1 >& point ) {
			// numerically stable one-pass update scheme
			if( num_points_ < std::numeric_limits<double>::epsilon() ) {
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
			else if( num_points_ < MAX_SURFEL_POINTS ) {
				const Eigen::Matrix< double, 6, 1 > deltaS = (mean_ - num_points_ * point);
				cov_ += 1.0 / (num_points_ * (num_points_ + 1.0)) * deltaS * deltaS.transpose();
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
		}

		inline void evaluate() {

			// determine effective view distance
			eff_view_dist_ = first_view_dir_.dot( initial_view_dir_ ) * first_view_inv_dist_;

			if( num_points_ >= MIN_SURFEL_POINTS ) {

				mean_ /= num_points_;
				cov_ /= (num_points_-1.0);

				// enforce symmetry..
				cov_( 1, 0 ) = cov_( 0, 1 );
				cov_( 2, 0 ) = cov_( 0, 2 );
				cov_( 3, 0 ) = cov_( 0, 3 );
				cov_( 4, 0 ) = cov_( 0, 4 );
				cov_( 5, 0 ) = cov_( 0, 5 );
				cov_( 2, 1 ) = cov_( 1, 2 );
				cov_( 2, 3 ) = cov_( 3, 2 );
				cov_( 2, 4 ) = cov_( 4, 2 );
				cov_( 2, 5 ) = cov_( 5, 2 );
				cov_( 3, 1 ) = cov_( 1, 3 );
				cov_( 3, 4 ) = cov_( 4, 3 );
				cov_( 3, 5 ) = cov_( 5, 3 );
				cov_( 4, 1 ) = cov_( 1, 4 );
				cov_( 4, 5 ) = cov_( 5, 4 );

				double det = cov_.block< 3, 3 >( 0, 0 ).determinant();

				if( det <= std::numeric_limits<double>::epsilon() ) {
					mean_.setZero();
					cov_.setZero();
					num_points_ = 0;
				}
				else {

					Eigen::Matrix< double, 3, 1 > eigen_values_;
					Eigen::Matrix< double, 3, 3 > eigen_vectors_;

					// eigen vectors are stored in the columns
					pcl::eigen33( Eigen::Matrix3d( cov_.block< 3, 3 >( 0, 0 ) ), eigen_vectors_, eigen_values_ );

					normal_ = eigen_vectors_.col( 0 );
					if( normal_.dot( first_view_dir_ ) > 0.0 )
						normal_ *= -1.0;

				}

			}

			up_to_date_ = true;
			unevaluated_ = false;

		}


		inline void unevaluate() {

			if( num_points_ > 0.0 ) {

				mean_ *= num_points_;
				cov_ *= (num_points_-1.0);

				unevaluated_ = true;

			}

		}


		Eigen::Matrix< double, 3, 1 > initial_view_dir_, first_view_dir_;

		float first_view_inv_dist_;
		float eff_view_dist_;
		double num_points_;

		Eigen::Matrix< double, 6, 1 > mean_;
		Eigen::Matrix< double, 3, 1 > normal_;
		Eigen::Matrix< double, 6, 6 > cov_;
		bool up_to_date_, applyUpdate_, unevaluated_;

		int idx_;

		float assocDist_;
		float assocWeight_;

		ShapeTextureFeature simple_shape_texture_features_;
		ShapeTextureFeature agglomerated_shape_texture_features_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};

	class NodeValue {
	public:
		NodeValue() {
			initialize();
		}

		NodeValue( unsigned int v ) {
			initialize();
		}

		~NodeValue() {
		}

		inline void initialize() {

			idx_ = -1;
			associated_ = 0;
			assocWeight_ = 1.f;
			border_ = false;

			surfels_[ 0 ].initial_view_dir_ = Eigen::Vector3d( 1., 0., 0. );
			surfels_[ 1 ].initial_view_dir_ = Eigen::Vector3d( -1., 0., 0. );
			surfels_[ 2 ].initial_view_dir_ = Eigen::Vector3d( 0., 1., 0. );
			surfels_[ 3 ].initial_view_dir_ = Eigen::Vector3d( 0., -1., 0. );
			surfels_[ 4 ].initial_view_dir_ = Eigen::Vector3d( 0., 0., 1. );
			surfels_[ 5 ].initial_view_dir_ = Eigen::Vector3d( 0., 0., -1. );

		}

		inline NodeValue& operator+=( const NodeValue& rhs ) {

			// merge surfels
			for( unsigned int i = 0; i < 6; i++ ) {

				Surfel& surfel = surfels_[ i ];

				if( surfel.applyUpdate_ ) {
					if( surfel.up_to_date_ )
						surfel.clear();

					surfel += rhs.surfels_[ i ];
				}

			}

			return *this;
		}

		inline Surfel* getSurfel( const Eigen::Vector3d& viewDirection ) {

			Surfel* bestMatchSurfel = NULL;
			double bestMatchDist = -1.;

			for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
				const double dist = viewDirection.dot( surfels_[ i ].initial_view_dir_ );
				if( dist > bestMatchDist ) {
					bestMatchSurfel = &surfels_[ i ];
					bestMatchDist = dist;
				}
			}

			return bestMatchSurfel;
		}

		inline void addSurfel( const Eigen::Vector3d& viewDirection, const Surfel& surfel ) {

			// find best matching surfel for the view direction
			Surfel* bestMatchSurfel = getSurfel( viewDirection );

			if( bestMatchSurfel->applyUpdate_ ) {
				if( bestMatchSurfel->up_to_date_ )
					bestMatchSurfel->clear();

				*bestMatchSurfel += surfel;
			}

		}

		inline void evaluateSurfels() {
			for( unsigned int i = 0; i < 6; i++ ) {
				if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {
					surfels_[ i ].evaluate();
				}
			}
		}


		inline void unevaluateSurfels() {
			for( unsigned int i = 0; i < 6; i++ ) {
				if( surfels_[i].up_to_date_ ) {
					surfels_[i].unevaluate();
				}
			}
		}


		Surfel surfels_[ 6 ];
		char associated_; // -1: disabled, 0: not associated, 1: associated, 2: not associated but neighbor of associated node
		spatialaggregate::OcTreeNode< float, NodeValue >* association_;
		char assocSurfelIdx_, assocSurfelDstIdx_;
		float assocWeight_;

		bool border_;

		int idx_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

	class ImagePreAllocator {
	public:
		ImagePreAllocator();

		~ImagePreAllocator();

		struct Info {
			Info() {
			}
			Info( NodeValue* v, const spatialaggregate::OcTreeKey< float, NodeValue >& k, unsigned int d )
					: value( v ), key( k ), depth( d ) {
			}

			NodeValue* value;
			spatialaggregate::OcTreeKey< float, NodeValue > key;
			unsigned int depth;
		};

		void prepare( unsigned int w, unsigned int h, bool buildNodeImage );

		spatialaggregate::DynamicAllocator< NodeValue > imageNodeAllocator_;
		uint64_t* imgKeys;
		NodeValue** valueMap;
		std::vector< Info > infoList;
		unsigned int width, height;
		spatialaggregate::OcTreeNode< float, NodeValue >** node_image_;
		std::set< spatialaggregate::OcTreeNode< float, NodeValue >* > node_set_;

	};

	MultiResolutionColorSurfelMap( float minResolution, float radius,
			boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator = boost::make_shared< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > >() );

	~MultiResolutionColorSurfelMap();

	void extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov );

	struct NodeSurfel {
		spatialaggregate::OcTreeNode< float, NodeValue >* node;
		Surfel* surfel;
	};

	static void convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta );
	static void convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b );

	void addPoints( const boost::shared_ptr< const pcl::PointCloud< pcl::PointXYZRGBA > >& cloud, const boost::shared_ptr< const std::vector< int > >& indices );

	void addPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices );

	void addImage( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, bool smoothViewDir = true, bool buildNodeImage = false );

	void getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition );

	static inline bool splitCriterion( spatialaggregate::OcTreeNode< float, NodeValue >* oldLeaf, spatialaggregate::OcTreeNode< float, NodeValue >* newLeaf );

	void findImageBorderPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, std::vector< int >& indices );

	void findVirtualBorderPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, std::vector< int >& indices );

	void findForegroundBorderPoints( const pcl::PointCloud<pcl::PointXYZRGBA>& cloud, std::vector< int >& indices );

	void clearAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices );

	void markNoUpdateAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices );

	void clearUpdateSurfelsAtPoints( const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, const std::vector< int >& indices );

	void markBorderAtPoints( const pcl::PointCloud<pcl::PointXYZRGBA>& cloud, const std::vector< int >& indices );

	static inline void clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	void clearBorderFlag();

	void markUpdateAllSurfels();
	static inline void markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );

	void markUpdateImprovedEffViewDistSurfels( const Eigen::Vector3f& viewPosition );
	static inline void markUpdateImprovedEffViewDistSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );

	void evaluateSurfels();
	void unevaluateSurfels();

	bool pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold );

	void setApplyUpdate( bool v );

	void setUpToDate( bool v );

	void clearUnstableSurfels();

	void buildShapeTextureFeatures();

	void clearAssociatedFlag();
	void distributeAssociatedFlag();

	void clearAssociationDist();

	void clearAssociations();
	static inline void clearAssociationsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );

	std::vector< unsigned int > findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud< pcl::PointXYZRGBA >& cloud, int maxDepth );

	void visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr, int depth, int viewDir, bool random = true );

	void visualizePrincipalSurface( pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloudPtr, int depth, int viewDir );
	bool projectOnPrincipalSurface( Eigen::Vector3d& sample, const std::vector< Surfel* >& neighbors,
				const std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& centerPositions, double resolution );

	static inline void evaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void unevaluateSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
	static inline void clearUnstableSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void setApplyUpdateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void setUpToDateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void evaluateSurfelPairRelationsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void clearAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void distributeAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void clearAssociationDistFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void buildSimpleShapeTextureFeatureFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void buildAgglomeratedShapeTextureFeatureFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void visualize3DColorDistributionFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
	static inline void visualizePrincipalSurfaceFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );

	void save( const std::string& filename );
	void load( const std::string& filename );

	boost::shared_ptr< spatialaggregate::OcTree< float, NodeValue > > octree_;
	boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator_;
	boost::shared_ptr< ImagePreAllocator > imageAllocator_;

	float min_resolution_, max_range_;

	static gsl_rng* r;

	pcl::StopWatch stopwatch_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};

class ShapeTextureTable {
public:

	ShapeTextureTable() {
		initialize();
	}
	~ShapeTextureTable() {
	}

	Eigen::Matrix< float, 1, NUM_SHAPE_BINS > shape_value_table_[ 3 ][ SHAPE_TEXTURE_TABLE_SIZE ];
	Eigen::Matrix< float, 1, NUM_TEXTURE_BINS > texture_value_table_[ 3 ][ SHAPE_TEXTURE_TABLE_SIZE ];

	void initialize();

	static ShapeTextureTable* instance() {
		if( !instance_ )
			instance_ = new ShapeTextureTable();
		return instance_;
	}

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

protected:
	static ShapeTextureTable* instance_;

};

}


#endif /* MULTIRESOLUTION_CSURFEL_MAP_H_ */

