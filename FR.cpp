#include <opencv2/opencv.hpp>
// #include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include "./faceRecLib/faceRec.hpp"

using namespace cv;

bool _openCamera(cv::VideoCapture capture) //success or fail to open cam.
{   
    /*
    函数名： _openCamera
    功能： 判断摄像头是否存在异常。
    输入参数：
    ----------------
        capture： VideoCapture
            摄像头相关参数
    返回值：
    ----------------
        bool
            真/假，当摄像头打开成功时返回真，否则假。
    */
    // sleep(3); //sleep 5 sec 
    try{
        if ( !capture.isOpened()){ //Exit when failing to open camera.
            // cout << "fail to open!" << endl;
            throw std::string("ERROR 103:fail to open camera!");
        }
        // 如果能获取到图像的分辨率信息则表明摄像头打开成功。
        // this device is ok if we can get the relate pars of video.
        long video_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        long video_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        if (video_h==0|| video_w==0){
            throw std::string("ERROR 104:camera has some situation!");            
        }
    }
    // 打印异常信息
    catch(std::string s){
        std::cout<< s << std::endl;
        return false;
    }
    return true;
}


void detectEye(std::vector<Rect> faces, cv::Mat frame)
{    
    //-------------detect eye-----------------------------
    cv::Mat hyframe;
    equalizeHist(frame, hyframe); 
    std::vector<Rect> eyes;
    const char *eyeCascadeFilename = "/usr/local/opencv-4.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load(eyeCascadeFilename);
    for(size_t i=0; i < faces.size(); i++)
    {
        Mat face_ = hyframe(faces[i]);
        eyeCascade.detectMultiScale(face_, eyes, 1.2, 2, 0, Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++)
        {
            // 在原图上标注眼睛，需要人脸在原图上的坐标
            Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            cv::circle(frame, eyeCenter, radius, Scalar(65, 105, 255), 4, 8, 0);
        }
    }
    //----------------------------------------------------
}


void detectEye(cv::Rect faces, cv::Mat frame)
{    
    //-------------detect eye-----------------------------
    cv::Mat hyframe;
    equalizeHist(frame, hyframe); 
    std::vector<Rect> eyes;
    const char *eyeCascadeFilename = "/usr/local/opencv-4.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load(eyeCascadeFilename);
    
    Mat face_ = hyframe(faces);
    eyeCascade.detectMultiScale(face_, eyes, 1.2, 2, 0, Size(30, 30));
    for (size_t j = 0; j < eyes.size(); j++)
    {
        // 在原图上标注眼睛，需要人脸在原图上的坐标
        Point eyeCenter(faces.x + eyes[j].x + eyes[j].width / 2, faces.y + eyes[j].y + eyes[j].height / 2);
        int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
        cv::circle(frame, eyeCenter, radius, Scalar(65, 105, 255), 4, 8, 0);
    }
    //----------------------------------------------------
}


int main()
{       

    FaceRec obj_fr;
    obj_fr.init_();

    /*整理数据库中的人脸数据*/
    cv::String pattern="/home/chenzl/chenzl/C++/faceRecognition/face_image/*";
    std::vector<cv::String> fn;
    cv::glob(pattern, fn, false);
    std::vector<Mat> people_face;
    std::vector<std::string> people_name;
    std::vector<std::string> filePath_vec;
    size_t count = fn.size(); //number of png files in images folder
    for (size_t i = 0; i < count; i++)
    {
        people_face.emplace_back(cv::imread(fn[i]));
        filePath_vec.push_back(fn[i]);
        people_name.emplace_back(obj_fr.GetPathOrURLShortName(fn[i]));        
    }

    /*open IP camera and read RTSP Stream*/
    cv::String rtsp = "rtsp://admin:IAOTWT@192.168.136.99:554/mpeg4/ch1/main/av_stream";
    // cv::String rtsp = "rtsp://admin:IAOTWT@192.168.136.99:554/mpeg4/ch1/main/av_stream";
    cv::VideoCapture capture(rtsp); 
    bool isgetframe;
    if (capture.isOpened() == false)
    {
        capture = cv::VideoCapture(rtsp);
        if(capture.isOpened() == false)
        {
            std::cout << "can not open cam!" <<std::endl;
            return 0;
        }
    }
    // capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920.0);  //设置摄像头采集图像分辨率
    // capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1440.0);

    
    //------------extract face feature information-----------------
    std::vector<dlib::matrix<float, 0, 1>> test_img_vec_tmp;
    std::vector<dlib::matrix<float, 0, 1>> test_img_vec;
    for(int i = 0 ;i < filePath_vec.size(); i++)
    {
        test_img_vec_tmp = obj_fr.extract_face_feature(filePath_vec[i]);
        test_img_vec.push_back(test_img_vec_tmp[0]);
    }    
    //--create a vector for saving face feature from frame of cam--
    std::vector<dlib::matrix<float, 0 ,1>> det_vec;

    bool flag = true;
    cv::Mat show_Img;
    std::vector<dlib::rectangle> tmp_dets;
    std::vector<std::string> tmp_names;
    cv::Mat frame, src;

    while(true){               
        isgetframe = capture.read(frame);
        if (isgetframe == false)
        {
            capture = cv::VideoCapture(rtsp);
            isgetframe = capture.read(frame);
            if(isgetframe == false)
            {
                std::cout << "can not open cam!" <<std::endl;
                return 0;
            }
        }
        // else {
        //     capture.read(frame);
        // }
        
        if(frame.empty())
        {
            continue ;
        }
        // mat -> dlib
        if (flag == true){


            dlib::matrix<dlib::bgr_pixel> img = obj_fr.MatToDlib(frame);        
            // dlib::matrix<dlib::rgb_pixel> img;
            // dlib::load_image(img, "1.png");
            std::vector<dlib::rectangle> dets = obj_fr.detetor(img);  
            if (dets.size() == 0)
            {                       
                std::cout << "no face" << std::endl;         
                cv::cvtColor(frame, frame , cv::COLOR_BGR2RGB);
                cv::imshow("win", frame);
                cv::waitKey(1);
            }
            else{ //dets.size() > 0
                // tmp_dets = dets;
                det_vec = obj_fr.extract_face_feature(dets, frame);
                bool isExitPeople = false;
                cv::Mat show_Img = dlib::toMat(img); //dlib->opencv   
                float error_min;
                //-----------plot the bounding box by opencv------------       
                for(int i=0; i<test_img_vec.size();i++)
                {
                    error_min = 100;
                    int tmp_index;
                    for(int j=0; j<det_vec.size();j++)
                    {
                        float error_val = obj_fr.cal_dist(test_img_vec[i],det_vec[j]);
                        if(error_val < error_min)
                        {
                            error_min = error_val;
                            tmp_index = j;
                        }
                        // else if(error_val > 0.75)
                        // {
                        //     show_Img = showImgAndBdBox(show_Img, dets[j], "unknown one");
                        // }                        
                    }
                    if(error_min < 0.40)
                    {
                        isExitPeople = true;
                        show_Img = obj_fr.showImgAndBdBox(show_Img, dets[tmp_index], people_name[i]);
                        std::cout << "similarity: "<< error_min <<std::endl;
                        // save current det index and people name index
                        tmp_dets.push_back(dets[tmp_index]);
                        tmp_names.push_back(people_name[i]);
                    }
                }                         
                if(isExitPeople == false)
                {
                    cv::imshow("win", frame);
                    cv::waitKey(1);
                }else
                {
                    cv::imshow("win", show_Img);
                    cv::waitKey(1);
                }
                
            }                              
            flag = !flag;
        }
        else
        {
            if(tmp_dets.size()>0)
            {
                for (int i = 0; i < tmp_dets.size(); i++)
                {
                    frame = obj_fr.showImgAndBdBox(frame, tmp_dets[i], tmp_names[i]);    
                    // detectEye(tmp_dets[i],frame);
                }                  
                cv::imshow("win",frame);
                cv::waitKey(1);       
                tmp_dets.clear();
                tmp_names.clear();
                // std::vector<dlib::rectangle>().swap(tmp_dets);
                // std::vector<int>().swap(name_index);
            }
            else
            {                   
                cv::imshow("win",frame);
                cv::waitKey(1);
            }
            flag = !flag;                        
        }                            

    }

    return 0;
}

