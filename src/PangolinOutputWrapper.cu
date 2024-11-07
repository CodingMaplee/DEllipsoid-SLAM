#include <VoxelUtilHashSDF.h>
#include <RayCastSDFUtil.h>
#include <MarchingCubesSDFUtil.h>

__global__ void convertMarchCubeToGLArray ( float3* march_cube, float3* dVertexArray, uchar3* dColourArray, uint3* dIndicesArray )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    if (isnan(march_cube[6 * x + 0].x)|| isnan(march_cube[6 * x + 0].y)||isnan(march_cube[6 * x + 0].z)||
//            isnan(march_cube[6 * x + 2].x)|| isnan(march_cube[6 * x + 2].y)||isnan(march_cube[6 * x + 2].z)||
//            isnan(march_cube[6 * x + 4].x)|| isnan(march_cube[6 * x + 4].y)||isnan(march_cube[6 * x + 4].z)||
//            isinf(march_cube[6 * x + 0].x)|| isinf(march_cube[6 * x + 0].y)||isinf(march_cube[6 * x + 0].z)||
//            isinf(march_cube[6 * x + 2].x)|| isinf(march_cube[6 * x + 2].y)||isinf(march_cube[6 * x + 2].z)||
//            isinf(march_cube[6 * x + 4].x)|| isinf(march_cube[6 * x + 4].y)||isinf(march_cube[6 * x + 4].z)||
//            isnan(march_cube[6 * x + 1].x)|| isnan(march_cube[6 * x + 1].y)||isnan(march_cube[6 * x + 1].z)||
//            isnan(march_cube[6 * x + 3].x)|| isnan(march_cube[6 * x + 3].y)||isnan(march_cube[6 * x + 3].z)||
//            isnan(march_cube[6 * x + 5].x)|| isnan(march_cube[6 * x + 5].y)||isnan(march_cube[6 * x + 5].z)||
//            isinf(march_cube[6 * x + 1].x)|| isinf(march_cube[6 * x + 1].y)||isinf(march_cube[6 * x + 1].z)||
//            isinf(march_cube[6 * x + 3].x)|| isinf(march_cube[6 * x + 3].y)||isinf(march_cube[6 * x + 3].z)||
//            isinf(march_cube[6 * x + 5].x)|| isinf(march_cube[6 * x + 5].y)||isinf(march_cube[6 * x + 5].z)
//            ) return;
    dVertexArray[3*x + 0] = make_float3 ( march_cube[6 * x + 0].x,march_cube[6 * x + 0].y,march_cube[6 * x + 0].z );
    dVertexArray[3*x + 1] = make_float3 ( march_cube[6 * x + 2].x,march_cube[6 * x + 2].y,march_cube[6 * x + 2].z );
    dVertexArray[3*x + 2] = make_float3 ( march_cube[6 * x + 4].x,march_cube[6 * x + 4].y,march_cube[6 * x + 4].z );

    if ( dColourArray != nullptr ) {
        dColourArray[3*x + 0].x = 255.f * march_cube[6 * x + 1].x;
        dColourArray[3*x + 0].y = 255.f * march_cube[6 * x + 1].y;
        dColourArray[3*x + 0].z = 255.f * march_cube[6 * x + 1].z;

        dColourArray[3*x + 1].x = 255.f * march_cube[6 * x + 3].x;
        dColourArray[3*x + 1].y = 255.f * march_cube[6 * x + 3].y;
        dColourArray[3*x + 1].z = 255.f * march_cube[6 * x + 3].z;

        dColourArray[3*x + 2].x = 255.f * march_cube[6 * x + 5].x;
        dColourArray[3*x + 2].y = 255.f * march_cube[6 * x + 5].y;
        dColourArray[3*x + 2].z = 255.f * march_cube[6 * x + 5].z;
    }

    if ( dIndicesArray != nullptr ) {
        dIndicesArray[x] = make_uint3 ( 3*x +0, 3*x + 1, 3*x + 2 );
    }
}

extern "C" void launch_convert_kernel ( float3* march_cube, float3* dVertexArray, uchar3* dColourArray, uint3* dIndicesArray,
                                        unsigned int size )
{
    dim3 block ( 64 );
    dim3 grid ( size / 64 );
    convertMarchCubeToGLArray<<< grid, block>>> ( march_cube,dVertexArray,dColourArray,dIndicesArray );
}





/*


__global__ void convertnodeToGLArray ( Vertex* objs, float3* dVertexArray, unsigned int size )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if ( dVertexArray != nullptr ) {
        dVertexArray[x]= make_float3(objs[x].node_xyz.x,objs[x].node_xyz.y,objs[x].node_xyz.z);
    }

}

extern  "C" void node_convert_kernel(Vertex* objs, float3* dVertexArray, unsigned int size)
{
    dim3 block ( 64 );
    dim3 grid ( size / 64 );
    convertnodeToGLArray<<< grid, block>>> ( objs, dVertexArray, size);
}*/
