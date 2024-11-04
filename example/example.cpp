#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <MainLoop.h>
#include <algorithm>
#include <chrono>
#include <GlobalAppState.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
# include<ctime>

void LoadDirectories(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                     std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps);

vector<string> split(const string& str, const string& delim){
    vector<string> res;
    if("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char * strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p){
        string s = p;  //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }
    return res;
}

map<string, int> readLabel(string file)
{
    map<string, int> myMap;
    ifstream readFile;
    readFile.open(file);
    while(!readFile.eof()){
        string temp;
        readFile>>temp;
        if(temp.empty())
        {
            break;
        }
        vector<string> tempstr = split(temp ,"=");
        string key = tempstr[0];
        string value1 = tempstr[1];
        int value  = atoi(value1.c_str());;
        myMap.insert(make_pair(key,value)); //将字符串转换为键值对
    }
    return myMap;
}

int main ( int argc, char** argv )
{

    std::string app_config_file = "";
    std::string vocabulary = "";
    std::string dataset_setting = "";
    std::string dataset_root = "";


    if ( argc != 5 )
    {
        app_config_file = std::string ("../zParametersDefault.txt");
        vocabulary = std::string ("../Vocabulary/ORBvoc.txt");
        dataset_setting = std::string ("../RGB-D/Bonn.yaml");
        dataset_root = std::string ("/home/AiLab/RGBD_DATASETS/rgbd_bonn_dataset/rgbd_bonn_placing_nonobstructing_box");
    }
    else{
        app_config_file = std::string ( argv[1] );
        vocabulary = std::string ( argv[2] );
        dataset_setting = std::string ( argv[3] );
        dataset_root = std::string ( argv[4] );
    }

    std::cout << "usage " << "app_config_file:" << app_config_file
            << " vocabulary:" << vocabulary
            << " dataset_setting:" << dataset_setting
            << " dataset_root:" << dataset_root << std::endl;

    std::string dataset_associate = dataset_root + "/associated.txt";
    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;

    LoadDirectories(dataset_associate, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    std::cout<<"LoadDirectories finish"<<std::endl;
    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    // Init all resources
    if ( !initSystem(vocabulary, dataset_setting, app_config_file))
    {
        std::cerr<<"Init failed, exit." << std::endl;
        return -1;
    }
    //
//    std::vector<string> labels;
//    ifstream readFile;
//    readFile.open(dataset_root + "/" + "labels.txt");
//    while(!readFile.eof())
//    {
//        string s;
//        getline(readFile,s);
//        if(!s.empty())
//        {
//            labels.push_back(s);
//        }
//        else{
//            break;
//        }
//    }

    //read label txt
    string file = dataset_root + "/" + "labels.txt";
    map<string, int> label_value = readLabel(file);


    // Main loop
    cv::Mat imRGB, imD, imMask;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(dataset_root+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED);//keep type
        imD = cv::imread(dataset_root+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //keep type
        ifstream f(dataset_root+"/mask/"+vstrImageFilenamesRGB[ni].substr(4,-1));
        if(f.good())
        {
            //imMask can be null
            imMask = cv::imread(dataset_root+"/mask/"+vstrImageFilenamesRGB[ni].substr(4,-1),cv::IMREAD_UNCHANGED);
        }
        double tframe = vTimestamps[ni];
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
        int width = imRGB.cols;
        int height = imRGB.rows;

        if (processInputRGBDFrame (imRGB, imD, imMask, label_value, tframe, vTimestamps,vTimesTrack))
        {
            std::cout<<"\tSuccess! frame " << vstrImageFilenamesRGB[ni] << " added into BundleFusion." << std::endl;
        }
        else
        {
            std::cout<<"\tFailed! frame " << vstrImageFilenamesRGB[ni] << " not added into BundleFusion." << std::endl;
        }
    }
    
    //while(cv::waitKey (20) != 'q');

    std::cout<<"deinit system"<<std::endl;
    deinitSystem();
    return 0;
}

void LoadDirectories(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                     vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}