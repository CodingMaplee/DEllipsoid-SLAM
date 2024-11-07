#pragma once
#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

#include <cuda_runtime.h>
#include "mLibCuda.h"
#include "VoxelUtilHashSDF.h"
#include "Eigen/Core"
#include "GlobalDefines.h"
class CUDAImageUtil {
public:
	template<class T> static void copy(T* d_output, T* d_input, unsigned int width, unsigned int height);
	//template<class T> static void resample(T* d_output, unsigned int outputWidth, unsigned int outputHeight, T* d_input, unsigned int inputWidth, unsigned int inputHeight);

    static void resampleToIntensity(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);

	static void resampleFloat4(float4* d_output, unsigned int outputWidth, unsigned int outputHeight, const float4* d_input, unsigned int inputWidth, unsigned int inputHeight);
	static void resampleFloat(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const float* d_input, unsigned int inputWidth, unsigned int inputHeight);
	static void resampleUCHAR4(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);

	static void convertDepthFloatToCameraSpaceFloat4(float4* d_output, const float* d_input, const Eigen::Matrix4f& intrinsicsInv, unsigned int width, unsigned int height);

	static void computeNormals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height);

	static void jointBilateralFilterColorUCHAR4(uchar4* d_output, uchar4* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

	static void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq);

	static void gaussFilterDepthMap(float* d_output, const float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
	//no invalid checks!
	static void gaussFilterIntensity(float* d_output, const float* d_input, float sigmaD, unsigned int width, unsigned int height);

	static void convertUCHAR4ToIntensityFloat(float* d_output, const uchar4* d_input, unsigned int width, unsigned int height);

	static void computeIntensityDerivatives(float2* d_output, const float* d_input, unsigned int width, unsigned int height);
	static void computeIntensityGradientMagnitude(float* d_output, const float* d_input, unsigned int width, unsigned int height);

	static void convertNormalsFloat4ToUCHAR4(uchar4* d_output, const float4* d_input, unsigned int width, unsigned int height);
	static void computeNormalsSobel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height);

	//adaptive filtering based on depth
	static void adaptiveGaussFilterDepthMap(float* d_output, const float* d_input, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height);
	static void adaptiveGaussFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float adaptFactor, unsigned int width, unsigned int height);

	static void jointBilateralFilterFloat(float* d_output, float* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
	static void adaptiveBilateralFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height);

	//static void undistort(float* d_depth, const mat3f& intrinsics, const float3& distortionParams, T defaultValue, const BaseImage<T>& noiseMask = BaseImage<T>())
    static void genPlaneMask(uint* planeMask, const float4* dots, float A, float B, float C, float D, uint width, uint height, Eigen::Matrix4f cam2world, float threshold);
    static void extractMotionConsistency(float* consistency_px, uchar* d_output, float threshold, const uchar* existingDynamicPx, const float* d_inputTar, const float* d_inputSrc, const float* d_intensityTar, const float* d_intensitySrc,
                                         const float* d_raftU, const float* d_raftV,const Eigen::Matrix4f& intrinsics,
                                         const Eigen::Matrix4f& transformLast, const Eigen::Matrix4f& transformCurrent_inv,
                                         unsigned int width, unsigned int height);
    static void genPersonMask(uchar* g_currMaskMapGpu, uchar*  personMask, uint g_maskMapWidth, uint g_maskMapHeight);
    static void int2uchar(int* mask, uchar* output, uint width, uint height);

    static void genObjectMask(uchar* maskGPUAll, uchar* maskGPU, uint* maskPixelNumGPU , uint maskIndex, uint width, uint height);
    static void GenerateRgb(uchar3* virtual_rgb, const Eigen::Vector4f boxBbox, const HashDataStruct& hashData, const Eigen::Matrix4f& camera_pose, const Eigen::Matrix4f& intrinsics ,uint width, uint height);

};

//TODO 
template<> void CUDAImageUtil::copy<float>(float*, float*, unsigned int, unsigned int);
template<> void CUDAImageUtil::copy<uchar4>(uchar4*, uchar4*, unsigned int, unsigned int);
//template void CUDAImageUtil::resample<float>(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int);
//template void CUDAImageUtil::resample<uchar4>(uchar4*, unsigned int, unsigned int, uchar4*, unsigned int, unsigned int);

#endif //CUDA_IMAGE_UTIL_H