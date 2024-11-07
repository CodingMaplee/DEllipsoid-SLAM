
#include <MainLoop.h>
#include <TimingLog.h>

#include <CUDAImageManager.h>

#include <DualGPU.h>

#include <TrajectoryManager.h>
#include <GlobalAppState.h>
#include <DepthSensing/TimingLogDepthSensing.h>
//#include <DepthSensing/Util.h>
#include <DepthSensing/CUDASceneRepHashSDF.h>
#include <DepthSensing/CUDARayCastSDF.h>
#include <DepthSensing/CUDAMarchingCubesHashSDF.h>
#include <DepthSensing/CUDAHistogramHashSDF.h>
#include <DepthSensing/CUDASceneRepChunkGrid.h>
#include <CUDAImageManager.h>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include "VisualizationHelper.h"
#include "KMeans/Kmeans.h"
#include "KMeans/Point3.h"
#include "DynamicObjectManager.h"
#include "ImageDetections.h"
#include "mLib.h"
#ifdef WITH_VISUALIZATION
#include <Output3DWrapper.h>
#include <PangolinOutputWrapper.h>
#endif
// Variables

ORB_SLAM3::System* SLAM = nullptr;
DynamicObjectManager* g_ObjectManager = nullptr;
ORB_SLAM3::ImageDetectionsManager* g_imageDetectionsManager = nullptr;
CUDAImageManager* g_imageManager = nullptr;
#ifdef WITH_VISUALIZATION
Visualization::Output3DWrapper * wrapper = nullptr;
#endif

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CUDASceneRepHashSDF*		g_sceneRep = NULL;
CUDARayCastSDF*				g_rayCast = NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;

DepthCameraParams			g_depthCameraParams;

//managed externally
int surface_read_count = 0;
bool publish_rgb = true;
bool publish_depth = true;
bool publish_virtual_rgb = true;
bool publish_feature = true;
bool publish_mesh = true;
bool publish_optImg=true;
bool publish_dynamic = true;
cv::Mat virtual_img;
// Functions
/**
 * debug function
 * get rgbdSensor, current default select PrimeSenseSensor
 * */

void calculateSDFMap(float* sdfMap, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void removeExistingDynamicPx ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void integrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f & transformation);//uint* existingDynamicPx
void deIntegrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void reintegrate(int currFrameNumber);
uchar3* Gen_image(cv::Mat rgb, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation, uchar* mask_gpu, uint width, uint height, int currFrameNumber);
void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename = "./scans/scan.ply", bool overwriteExistingFile = false );
void StopScanningAndExit ( bool aborted = false );
void Gen_Virtual_image(const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation, Eigen::Vector4f Bbox, uint width, uint height, int currFrameNumber);

void ResetDepthSensing();

bool CreateDevice();
extern "C" void convertColorFloat4ToUCHAR4 ( uchar4* d_output, float4* d_input, unsigned int width, unsigned int height );

/*************************BundleFusion SDK Interface ********************/
bool initSystem (std::string vocFile, std::string settingFile, std::string app_config)
{
    GlobalAppState::getInstance().readMembers ( app_config );
    SLAM = new ORB_SLAM3::System(vocFile, settingFile, ORB_SLAM3::System::RGBD, true);
    Eigen::Matrix4f intrinsics = SLAM->GetDepthIntrinsics();
    float imageScale = SLAM->GetImageScale();
    uint imageWidth, imageHeight;
    SLAM->GetImageSize(imageWidth, imageHeight);
    try {
        g_imageDetectionsManager = new ORB_SLAM3::ImageDetectionsManager();
        g_ObjectManager = new DynamicObjectManager();
        g_imageManager = new CUDAImageManager ( GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
                                                imageWidth, imageHeight, intrinsics, imageScale, false );

#ifdef WITH_VISUALIZATION
        wrapper = new Visualization::PangolinOutputWrapper ( GlobalAppState::get().s_integrationWidth,GlobalAppState::get().s_integrationHeight );
#endif

        if ( !CreateDevice() )
        {
            std::cerr<<"Create Device failed. " << std::endl;
            return false;
        }
    }
    catch ( const std::exception& e )
    {
        //MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
        std::cerr<< ( "Exception caught" ) << std::endl;
        return false;
    }
    catch ( ... )
    {
        //MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
        std::cerr<< ( "UNKNOWN EXCEPTION" ) << std::endl;;
        return false;
    }
    return true;
}

cv::Mat ucharToMat(const uchar4* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    cv::Mat img(cv::Size(img_width, img_height), CV_8UC3);
    for (int i = 0; i < img_width * img_height; i++)
    {
        int b = p2[i].x;
        int g = p2[i].y;
        int r = p2[i].z;

        img.at<cv::Vec3b>(i / img_width, (i % img_width))[0] = r;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[1] = g;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[2] = b;


    }
    return img;
}

uchar4* uchar3ToUchar4(const uchar3* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    uchar4 * img =  new uchar4[img_width*img_height];
    //cv::Mat img(cv::Size(img_width, img_height), CV_8UC3);
    for (int i = 0; i < img_width * img_height; i++)
    {
        int b = p2[i].x;
        int g = p2[i].y;
        int r = p2[i].z;

        img[i].x = b;
        img[i].y = g;
        img[i].z = r;
        img[i].w = 1.0;
    }
    return img;
}

cv::Mat ucharToMat(const uchar3* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    cv::Mat img(cv::Size(img_width, img_height), CV_8UC3);
    for (int i = 0; i < img_width * img_height; i++)
    {
        int b = p2[i].x;
        int g = p2[i].y;
        int r = p2[i].z;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[0] = r;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[1] = g;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[2] = b;
    }
    return img;
}

uchar3* MatToUchar(const cv::Mat img, const int width, const int height)
{

    uchar3 *img_cpu = new uchar3[width * height];
    int img_width = width;
    int img_height = height;
    for (int i = 0; i < img_width * img_height; i++)
    {
         img_cpu[i].z = img.at<cv::Vec3b>(i / img_width, (i % img_width))[0];
         img_cpu[i].y = img.at<cv::Vec3b>(i / img_width, (i % img_width))[1];
        img_cpu[i].x = img.at<cv::Vec3b>(i / img_width, (i % img_width))[2];
    }
    return img_cpu;
}

uchar4* MatToUchar4(const cv::Mat img, const int width, const int height)
{

    uchar4 *img_cpu = new uchar4[width * height];
    int img_width = width;
    int img_height = height;
    for (int i = 0; i < img_width * img_height; i++)
    {
        img_cpu[i].z = img.at<cv::Vec3b>(i / img_width, (i % img_width))[0];
        img_cpu[i].y = img.at<cv::Vec3b>(i / img_width, (i % img_width))[1];
        img_cpu[i].x = img.at<cv::Vec3b>(i / img_width, (i % img_width))[2];
        img_cpu[i].w = 1.0;
    }
    return img_cpu;
}

cv::Mat floatToMat(const float* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    cv::Mat img(cv::Size(img_width, img_height), CV_32FC1);
    for (int i = 0; i < img_width * img_height; i++)
    {
        float d = p2[i];

        img.at<float>(i / img_width, (i % img_width)) = d;
    }
    return img;
}

bool processInputRGBDFrame (cv::Mat& rgb, cv::Mat& depth, cv::Mat& mask, map<string, int> label_value, double tframe, vector<double> vTimestamps, vector<float>& vTimesTrack)
{
    //pre-process
    cv::Mat depthClone = depth.clone();
    cv::Mat depthShow = depth.clone();
    uchar4 *  d_virtual_Rgb = new uchar4[640 * 480];
    cv::Mat VirtualRgb (cv::Size(1280, 960), CV_8UC3);
    cv::Mat featureImg;
    depthClone.convertTo(depthClone,CV_32F);
    cv::Mat depthFiltered = depthClone.clone();
    cv::bilateralFilter(depthClone, depthFiltered, 5, 1.0, 1.0);//, cv::BORDER_DEFAULT
    // Read Input
    ///////////////////////////////////////
    double depthFactor = SLAM->GetDepthFactor();
    bool bGotDepth = g_imageManager->process ( rgb, depthFiltered, depthFactor );
    uint currFrameNumber = g_imageManager->getCurrFrameNumber();
    std::cout<<"currFrameNumber: "<<currFrameNumber<<endl;
    uint width = g_imageManager->getIntegrationWidth();
    uint height = g_imageManager->getIntegrationHeight();
    Eigen::Matrix4f intrinsics = g_imageManager->getDepthIntrinsics();
    int person_id = label_value.find("person")->second;
    int box_id = label_value.find("bottle")->second;
    cout<<"box_id:"<<box_id<<endl;
    int chair1_id = label_value.find("chair1")->second;
    SLAM->SetMapLabelValue(label_value);
    vector<ORB_SLAM3::Detection::Ptr> detections;
    uchar* boxMask = new uchar[height * width];
    //all maskData
    uchar* maskData = new uchar[width * height];
    maskData = (uchar*) mask.data;
    if (bGotDepth) {

        uchar* maskAll = new uchar[width * height];
        maskAll = (uchar*) mask.data;
        uint pixelNum;
        int object_num = label_value.size();
        for (int i = 1; i <= object_num; i++) {
            uint4 box;
            box.x = 640;
            box.y = 480;
            box.z = 0;
            box.w = 0;
            pixelNum=0;
            uchar *maskCpu = new uchar[width * height];
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    if (maskAll[w + h * width] == i) {

                        maskCpu[w + h * width] = 255;
                        pixelNum++;
                        if (w < box.x) { box.x = w; }
                        if (h < box.y) { box.y = h; }
                        if (w > box.z) { box.z = w; }
                        if (h > box.w) { box.w = h; }
                    } else {
                        maskCpu[w + h * width] = 0;
                    }
                }
            }
            if (pixelNum > 0) {
                ORB_SLAM3::Detection *det = new ORB_SLAM3::Detection();
                det->bbox = Eigen::Vector4d(box.x, box.y, box.z, box.w);
                det->mask = maskCpu;
                det->category_id = i;
                unordered_set<int>objectIdInMap = SLAM->ObjectInMap();

                if (objectIdInMap.find(i)!= objectIdInMap.end()){
                    bool CenterPoint_true = det->GenCenterPoint(depthFiltered, intrinsics);
                    detections.push_back(std::shared_ptr<ORB_SLAM3::Detection>(det));
                }
                //|| i== box_id
                else if (det->GenCenterPoint(depthFiltered, intrinsics)  || i == person_id)
                {
                    detections.push_back(std::shared_ptr<ORB_SLAM3::Detection>(det));
                }
            }
        }
       cv::Mat instanceMat = cv::Mat(height, width, CV_8UC1);
        cv::Mat instanceMat_box = cv::Mat(height, width, CV_8UC1);
        cv::Mat instanceMat_chair1 = cv::Mat(height, width, CV_8UC1);

        instanceMat = cv::Scalar(0);
        instanceMat_box = cv::Scalar(0);
        for (int k=0; k<detections.size(); k++){
            if (detections[k]->category_id == person_id ){
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        if (detections[k]->mask[width * i + j] == 255) {
                            instanceMat.at<uchar>(i, j) = 255;
                        }

                    }
                }
            }
            //and
            if (detections[k]->category_id ==  box_id || detections[k]->category_id ==  person_id ){
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        if (detections[k]->mask[width * i + j] == 255) {
                            instanceMat_box.at<uchar>(i, j) = 255;
                        }
                    }
                }
            }
        }
        //cv::imwrite("../chair_mat/" + std::to_string(currFrameNumber) + ".png", instanceMat_chair1);
        cv::Mat se1 = cv::getStructuringElement(0, cv::Size(50,50));
        cv::Mat se2 = cv::getStructuringElement(0, cv::Size(40,40));
        cv::dilate(instanceMat, instanceMat, se1, cv::Point(-1, -1), 1);
        cv::dilate(instanceMat_box, instanceMat_box, se2, cv::Point(-1, -1), 1);
        uchar* personMask = new uchar[height * width];
        personMask = (uchar*)instanceMat.data;
        boxMask = (uchar*)instanceMat_box.data;
        g_imageManager->setPersonMap(personMask);
        if (currFrameNumber==0){
            SLAM->SetOptImg(rgb);
        }
        if (currFrameNumber>0){
            SLAM->SetOptImg(virtual_img);
            //cv::imwrite("../box_mat/" + std::to_string(currFrameNumber) + ".png",  SLAM->GetOptImg());
        }

        SLAM->TrackRGBD(rgb, depthFiltered, detections, featureImg, personMask, tframe);
        //cv::imwrite("../img/" + std::to_string(currFrameNumber) + ".png",  featureImg);
        std::cout << "tracking!" << std::endl;
        ///////////////////////////////////////
        // Fix old frames, and fuse the segment result to the TSDF model.
        ///////////////////////////////////////
        //printf("start reintegrate\n");
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        //   reintegrate(currFrameNumber);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    }
    ///////////////////////////////////////
    // Reconstruction of current frame
    ///////////////////////////////////////
    if ( bGotDepth ) {
        std::vector<Sophus::SE3f> trajectories = SLAM->ExtractTrajectoryTUM();
        if (trajectories.size() == 0) {
            return;
        }
        Eigen::Matrix4f transformCurrent = trajectories[trajectories.size() - 1].matrix();
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        if (GlobalAppState::get().s_reconstructionEnabled) {
            uchar* mask_gpu;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&mask_gpu, sizeof(uchar) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(mask_gpu, maskData, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
            uchar* delete_gpu;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&delete_gpu, sizeof(uchar) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(delete_gpu, boxMask, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));

            DepthCameraData depthCameraData(g_imageManager->getIntegrateFrame(currFrameNumber).getDepthFrameGPU(),
                                                g_imageManager->getIntegrateFrame(currFrameNumber).getColorFrameGPU(),
                                                mask_gpu, delete_gpu);
            depthCameraData.d_currFrameNumber =currFrameNumber;
            integrate(depthCameraData, transformCurrent);
            uchar3 *virtual_RgbCpu  = Gen_image(rgb, depthCameraData, transformCurrent, boxMask, width, height, currFrameNumber);
            //cout<<"33333333333333"<<endl;
            virtual_img = ucharToMat(virtual_RgbCpu,640,480);
            d_virtual_Rgb = MatToUchar4(SLAM->GetOptImg(),640,480);
            //cout<<"444444444444444"<<endl;
        }
    }

#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        const uchar4* d_color = g_imageManager->getLastIntegrateFrame().getColorFrameCPU();
        uchar4* d_feature = MatToUchar4(featureImg,width,height);
//        cv::Mat optImg = SLAM->GetOptImg();
//        uchar4* d_opt = MatToUchar4(optImg,width,height);
        if ( publish_rgb )
        {
            wrapper->publishColorMap (d_feature);
        }

        if ( publish_feature )
        {
            wrapper->publishFeatureMap(d_virtual_Rgb);
        }

//        if (publish_optImg){
//            wrapper->publishOptMap(d_opt);
//        }
//
//        if ( publish_virtual_rgb )
//        {
//            wrapper->publishVirtualMap(d_virtual_Rgb);
//        }
        if ( publish_mesh )//
        {
            // surface get
            MarchingCubesData* march_cube = nullptr;

            if ( surface_read_count == 1 )
            {
                surface_read_count = 0;


                Timer t;

                march_cube = g_marchingCubesHashSDF->extractIsoSurfaceGPUNoChrunk ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );

                std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;
            }
            else
            {
                surface_read_count++;
            }

            if ( march_cube != nullptr )
            {
                wrapper->publishSurface ( march_cube );
            }
        }
    }
#endif

    return true;
}

void setPublishRGBFlag ( bool publish_flag )
{
    publish_rgb = publish_flag;
}

void setPublishMeshFlag ( bool publish_flag )
{
    publish_mesh = publish_flag;
}

bool saveMeshIntoFile ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{
    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();

    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );

    }
    else
    {
        vec4f posWorld = vec4f ( GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    return true;

}



bool deinitSystem()
{

    SAFE_DELETE ( g_sceneRep );
    SAFE_DELETE ( g_rayCast );
    SAFE_DELETE ( g_marchingCubesHashSDF );
    SAFE_DELETE ( g_historgram );
    SAFE_DELETE ( g_chunkGrid );


    SAFE_DELETE ( g_imageManager );


#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        wrapper->noticeFinishFlag();
    }
#endif

    SLAM->Shutdown();
//
//    // Tracking time statistics
////    sort(vTimesTrack.begin(),vTimesTrack.end());
////    float totaltime = 0;
////    for(int ni=0; ni<nImages; ni++)
////    {
////        totaltime+=vTimesTrack[ni];
////    }
////    cout << "-------" << endl << endl;
////    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
////    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    //SLAM->SaveKeyFrameInstance_result();
    SLAM->SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    return true;
}

/************************************************************************/



bool CreateDevice()
{

    g_sceneRep = new CUDASceneRepHashSDF ( CUDASceneRepHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    //g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_imageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv()));
    g_rayCast = new CUDARayCastSDF ( CUDARayCastSDF::parametersFromGlobalAppState ( GlobalAppState::get(), g_imageManager->getDepthIntrinsics(), g_imageManager->getDepthIntrinsicsInv() ) );

    g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF ( CUDAMarchingCubesHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    g_historgram = new CUDAHistrogramHashSDF ( g_sceneRep->getHashParams() );

    if ( GlobalAppState::get().s_streamingEnabled )
    {
        g_chunkGrid = new CUDASceneRepChunkGrid ( g_sceneRep,
                                                  GlobalAppState::get().s_streamingVoxelExtents,
                                                  GlobalAppState::get().s_streamingGridDimensions,
                                                  GlobalAppState::get().s_streamingMinGridPos,
                                                  GlobalAppState::get().s_streamingInitialChunkListSize,
                                                  GlobalAppState::get().s_streamingEnabled,
                                                  GlobalAppState::get().s_streamingOutParts );
    }
    if ( !GlobalAppState::get().s_reconstructionEnabled )
    {
        GlobalAppState::get().s_RenderMode = 2;
    }
    g_depthCameraParams.fx = g_imageManager->getDepthIntrinsics() ( 0, 0 ); //TODO check intrinsics
    g_depthCameraParams.fy = g_imageManager->getDepthIntrinsics() ( 1, 1 );

    g_depthCameraParams.mx = g_imageManager->getDepthIntrinsics() ( 0, 2 );
    g_depthCameraParams.my = g_imageManager->getDepthIntrinsics() ( 1, 2 );
    g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
    g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
    g_depthCameraParams.m_imageWidth = g_imageManager->getIntegrationWidth();
    g_depthCameraParams.m_imageHeight = g_imageManager->getIntegrationHeight();
    //std::cout<<g_depthCameraParams.fx << "," << g_depthCameraParams.fy << "," <<g_depthCameraParams.mx << "," <<g_depthCameraParams.my << "," <<g_depthCameraParams.m_sensorDepthWorldMin << "," <<g_depthCameraParams.m_sensorDepthWorldMax << "," <<g_depthCameraParams.m_imageWidth << "," <<g_depthCameraParams.m_imageHeight << std::endl;
    DepthCameraData::updateParams ( g_depthCameraParams );

    //std::vector<DXGI_FORMAT> rtfFormat;
    //rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM); // _SRGB
    //V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight, rtfFormat));

    //g_CudaImageManager->OnD3D11CreateDevice(pd3dDevice);

    return true;
}

uchar3 * Gen_image(cv::Mat rgb, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation, uchar* mask_cpu, uint width, uint height, int currFrameNumber) {
    //uchar3 *img_gpu;
    //MLIB_CUDA_SAFE_CALL(cudaMalloc(&img_gpu, sizeof(uchar3) * width * height));
    uchar3* virtualRgb_cpu = MatToUchar(rgb, width, height);
    uchar3* virtualRgb_gpu;
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&virtualRgb_gpu, sizeof(uchar3) * width * height));
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(virtualRgb_gpu, virtualRgb_cpu, sizeof(uchar3) * width * height, cudaMemcpyHostToDevice));
    if ( GlobalAppState::get().s_streamingEnabled ){
        Eigen::Vector4f trans = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                 GlobalAppState::getInstance().s_streamingPos.y,
                                                                 GlobalAppState::getInstance().s_streamingPos.z,
                                                                 1.0);
        vec4f posWorld = vec4f(trans(0), trans(1), trans(2), trans(3)); // trans laggs one frame *trans
        vec3f p(posWorld.x, posWorld.y, posWorld.z);
        g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, true, true);
        g_chunkGrid->streamInToGPUPass1GPU(true);
    }
    if ( GlobalAppState::get().s_integrationEnabled ){
        unsigned int *d_bitMask = NULL;
        if (g_chunkGrid) d_bitMask = g_chunkGrid->getBitMaskGPU();
        uchar* mask_gpu;
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&mask_gpu, sizeof(uchar) * width * height));
        MLIB_CUDA_SAFE_CALL(cudaMemcpy(mask_gpu, mask_cpu, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
        g_sceneRep->gen_image(virtualRgb_gpu, transformation, depthCameraData, g_depthCameraParams, mask_gpu, d_bitMask);
        uchar3 *img_cpu = new uchar3[width * height];
        MLIB_CUDA_SAFE_CALL(cudaMemcpy(img_cpu, virtualRgb_gpu, sizeof(uchar3) * width * height, cudaMemcpyDeviceToHost));
        MLIB_CUDA_SAFE_CALL(cudaFree(virtualRgb_gpu));
        cv::Mat imgMat = ucharToMat(img_cpu,width,height);
        cv::GaussianBlur(imgMat,imgMat,cv::Size(9,9),1);
        return  img_cpu;
    }
}
void removeExistingDynamicPx( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                 GlobalAppState::getInstance().s_streamingPos.y,
                                                                 GlobalAppState::getInstance().s_streamingPos.z, 1.0);
        vec4f posWorld = vec4f ( trans(0), trans(1), trans(2), trans(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }
    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid )
            d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->removeExistingDynamicPx ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);


    }
}
void calculateSDFMap(float* sdfMap, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                 GlobalAppState::getInstance().s_streamingPos.y,
                                                                 GlobalAppState::getInstance().s_streamingPos.z, 1.0);
        vec4f posWorld = vec4f ( trans(0), trans(1), trans(2), trans(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }
    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid )
            d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->calculateSDFMap ( sdfMap, transformation, depthCameraData, g_depthCameraParams, d_bitMask);

    }
}
void integrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)//uint* existingDynamicPx
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans_posWorld = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                          GlobalAppState::getInstance().s_streamingPos.y,
                                                                          GlobalAppState::getInstance().s_streamingPos.z, 1.0f);
        vec4f posWorld = vec4f ( trans_posWorld(0), trans_posWorld(1), trans_posWorld(2), trans_posWorld(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();

        g_sceneRep->integrate ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);


    }
}

void deIntegrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans_posWorld = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                          GlobalAppState::getInstance().s_streamingPos.y,
                                                                          GlobalAppState::getInstance().s_streamingPos.z, 1.0f);
        vec4f posWorld = vec4f ( trans_posWorld(0), trans_posWorld(1), trans_posWorld(2), trans_posWorld(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->deIntegrate ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);
    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}



void reintegrate( int currFrameNumber)
{
    //find the 10(maxPerFrameFixes) frames with the biggest transform, and the frames with the sementic information
//    const unsigned int maxPerFrameFixes = GlobalAppState::get().s_maxFrameFixes;
//    TrajectoryManager* tm = g_bundler->getTrajectoryManager();
//    //std::cout << "reintegrate():" << tm->getNumActiveOperations() << " : " << tm->getNumOptimizedFrames() << std::endl;
//    uint width = g_imageManager->getIntegrationWidth();
//    uint height = g_imageManager->getIntegrationHeight();
//    if ( tm->getNumActiveOperations() < maxPerFrameFixes)
//    {
//        tm->generateUpdateLists();
//        //if (GlobalBundlingState::get().s_verbose) {
//        //	if (tm->getNumActiveOperations() == 0)
//        //		std::cout << __FUNCTION__ << " :  no more work (everything is reintegrated)" << std::endl;
//        //}
//    }
//
//    for ( unsigned int fixes = 0; fixes < maxPerFrameFixes; fixes++ )
//    {
//        mat4f newTransform = mat4f::zero();
//        mat4f oldTransform = mat4f::zero();
//        unsigned int frameIdx = ( unsigned int )-1;
//
//
//        if ( tm->getTopFromDeIntegrateList ( oldTransform, frameIdx ) )
//        {
//
////          DepthCameraData depthCameraData( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicMaskFrameGPU() );
//            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && !isnan ( newTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            //deIntegrate ( depthCameraData, oldTransform);
//            MLIB_CUDA_SAFE_CALL(cudaFree(bbox_gpu));
//
//            continue;
//        }
//        else if ( tm->getTopFromIntegrateList ( newTransform, frameIdx ) )
//        {
//            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
//            MLIB_ASSERT ( !isnan ( newTransform[0] ) && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            //integrate ( depthCameraDataErode, newTransform);
//            tm->confirmIntegration ( frameIdx );
//            continue;
//        }
//        else if ( tm->getTopFromReIntegrateList ( oldTransform, newTransform, frameIdx ) )
//        {
//            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
//            DepthCameraData depthCameraData( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicBoxFrameGPU() );
//            DepthCameraData depthCameraDataErode( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicBoxErodeFrameGPU() );
//            //uint *DynamicBoxFrameCpu = new  uint[width * height];
//            //MLIB_CUDA_SAFE_CALL(cudaMemcpy(DynamicBoxFrameCpu,f.getDynamicBoxFrameGPU(),sizeof(uint)*width * height, cudaMemcpyDeviceToHost));
//            //VisualizationHelper::ShowUint(DynamicBoxFrameCpu, width, height,"/home/user/2022AI/Large-scale_dynamic_scene_reconstruction/DynamicBoxFrame/" + std::to_string(currFrameNumber)+"---"+std::to_string(frameIdx));
//            //uint *DynamicBoxErodeFrameCpu = new  uint[width * height];
//            //MLIB_CUDA_SAFE_CALL(cudaMemcpy(DynamicBoxErodeFrameCpu,f.getDynamicBoxErodeFrameGPU(),sizeof(uint)*width * height, cudaMemcpyDeviceToHost));
//            //VisualizationHelper::ShowUint(DynamicBoxErodeFrameCpu, width, height,"/home/user/2022AI/Large-scale_dynamic_scene_reconstruction/DynamicBoxErodeFrame/"+ std::to_string(currFrameNumber)+"---"+std::to_string(frameIdx));
//            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && !isnan ( newTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            deIntegrate ( depthCameraData, oldTransform);
//            integrate ( depthCameraDataErode, newTransform);
//            tm->confirmIntegration ( frameIdx );
//            continue;
//        }
//        else
//        {
//            break; //no more work to do
//        }
//    }
//    g_sceneRep->garbageCollect();
}

void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{

    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );
        //g_chunkGrid->startMultiThreading();
    }
    else
    {
        vec4f posWorld = vec4f ( GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}

void ResetDepthSensing()
{
    g_sceneRep->reset();
    //g_Camera.Reset();
    if ( g_chunkGrid )
    {
        g_chunkGrid->reset();
    }
}


