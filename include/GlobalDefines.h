
#pragma once
#ifndef _GLOBAL_DEFINES_
#define _GLOBAL_DEFINES_

#define MINF __int_as_float(0xff800000)

#define MAX_MATCHES_PER_IMAGE_PAIR_RAW 128
#define MAX_MATCHES_PER_IMAGE_PAIR_FILTERED 25

/////////////////dynamic//////////////////
#define NODE_KNN 8
#define VERTEX_KNN 4
#define VERTICES_RADIUS 0.01f
#define NODES_RADIUS 0.08f
#define CONSISTENCY_THRESHOLD 6.0f
#define VERTEX_DOWN_SAMPLE 4
#define CONSISTENCY_MIN_NUM 50
#define CONCAVITY_MIN 0.02
#define CONCAVITY_MAX 0.07
#define IS_FILTER_SDF_RESIDUAL false
#define EXTEND_KEYFRAMES_DYNAMIC_REGION true
#define OPTIMIZED_TIMES 20

#define PERSON_CAT 1
#define PERSON_CAT2 2
#define OBJECT_CAT_NUM 7
#define BOX_CAT -1
#define DYNAMIC_BOX_DILATATION 40
#define DYNAMIC_BOX_ERODE 10
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define RECONSTRUCT_GRAY false
# define DYNAMIC_PERSON_POINT_NUMBER 20

#define USE_LIE_SPACE

#endif //_GLOBAL_DEFINES_
