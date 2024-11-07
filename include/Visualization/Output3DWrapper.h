#pragma once

#include <vector_types.h>
#include <MarchingCubesSDFUtil.h>

namespace Visualization{
  class Output3DWrapper{
  public:
	virtual ~Output3DWrapper(){};
	
	virtual void publishSurface(const MarchingCubesData* m_cube) = 0;

    virtual void publishObjCloud ( vector<Eigen::Vector4f> vertices) = 0;

    virtual void publishDynamic ( vector<float3x2*> vertices, vector<uint> vertex_num, vector<float3*> bbox ) = 0;

	virtual void publishColorMap(uchar4* rgba) = 0;

    virtual void publishVirtualMap(uchar4* rgba) = 0;

    virtual void publishFeatureMap(uchar4* rgba) = 0;

    virtual void publishOptMap(uchar4* rgba) = 0;

	virtual void publishColorRayCastedMap(float4* rgba) = 0;
	
	virtual void publishDepthRayCastedMap(float* depth) = 0;
	
	virtual void publishAllTrajetory(float* trajs, int size) = 0;
	
	virtual void publishCurrentCameraPose(float* pose) = 0;
	
	virtual void noticeFinishFlag() = 0;
	
	virtual bool getPublishRGBFlag() = 0;
	
	virtual bool getPublishMeshFlag() = 0;
  };
}