#include "faceRec.hpp"
using namespace dlib;
/*-----------------------------------------------------------------------------------*/
    //下一位代码定义Resnet网络。基本上是复制的
    //并从dnn_imagenet_ex.cpp示例粘贴，但我们替换了损失
    //使用损耗度量进行分层，使网络变得更小。去读导论吧
    //dlib dnn示例了解所有这些内容的含义。
    //另外，dnn_mtric_learning_on_images_ex.cpp示例显示了如何训练此网络。
    //本例使用的dlib_face_recognition_resnet_model_v1模型
    //基本上是dnn_metric_learning_on_images_ex.cpp中显示的代码，除了
    //小批量大（35x15而不是5x5），迭代没有进展
    //设置为10000，训练数据集由大约300万个图像组成，而不是
    //55。此外，输入层被锁定为150大小的图像。
    /*------------------------------------------------------------------------------------*/
    template <template <int, template<typename>class, int, typename>  class block, int N, template<typename>class BN, typename SUBNET>
    using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;
    
    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;
    
    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
    
    template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
    
    template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
    
    using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
        alevel0<
        alevel1<
        alevel2<
        alevel3<
        alevel4<
        dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
        dlib::input_rgb_image_sized<150>
        >>>>>>>>>>>>;




// std::string FaceRec::resnet_model_fileName = "/home/chenzl/chenzl/dlib/dlib_face_recognition_resnet_model_v1.dat";
// std::string FaceRec::face_shape_discriptor_fileName = "/home/chenzl/chenzl/dlib/shape_predictor_5_face_landmarks.dat";
// dlib::frontal_face_detector FaceRec::detetor = dlib::get_frontal_face_detector();
// dlib::deserialize(FaceRec::resnet_model_fileName) >> FaceRec::net;
// dlib::deserialize(FaceRec::face_shape_discriptor_fileName) >> FaceRec::sp;
void FaceRec::init_()
{   
    this->resnet_model_fileName = "/home/chenzl/chenzl/dlib/dlib_face_recognition_resnet_model_v1.dat";
    this->face_shape_discriptor_fileName = "/home/chenzl/chenzl/dlib/shape_predictor_5_face_landmarks.dat";
    this->detetor = dlib::get_frontal_face_detector();
    dlib::deserialize(FaceRec::resnet_model_fileName) >> this->net;
    dlib::deserialize(FaceRec::face_shape_discriptor_fileName) >> this->sp;
}


//  opencv mat -> dlib matrix
dlib::matrix<dlib::bgr_pixel> FaceRec::MatToDlib(cv::Mat cv_Img){
    cv::cvtColor(cv_Img, cv_Img, cv::COLOR_BGR2RGB);
    dlib::matrix<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<dlib::rgb_pixel>(cv_Img));
    return img;
}


/*-------------------extract 128D face feature in database-------------------------
@param 
1) dlib::frontal_face_detector-------the detector
2) dlib::shape_predictor-------------provide all feature points position for resnet
3) anet_type-------------------------resnet for extracting face feature according shape_predictor
4) std::string-----------------------people imgs path
@return 
1) std::vector<matrix<float,0,1>>----each matrix is a face feature for relevant people*/
std::vector<dlib::matrix<float, 0, 1>> FaceRec::extract_face_feature(const std::string& filePath){
    
    dlib::matrix<dlib::rgb_pixel> test_img;    
    //----------------preload people image------------------------
    dlib::load_image(test_img, filePath);
    //--------------detect face part in people image--------------
    std::vector<dlib::rectangle> dets_test = this->detetor(test_img);
    //---create a rgb_pixel to save people face part image--------
    dlib::matrix<dlib::rgb_pixel> test_face_chip;    
    auto test_shape = FaceRec::sp(test_img, dets_test[0]); //get the sp.dat position information
    //---extract face part and save to test_face_chip-------------
    dlib::extract_image_chip(test_img, dlib::get_face_chip_details(test_shape,150,0.25), test_face_chip);
    //---push back the above people`s face img to faces set-------
    std::vector<dlib::matrix<dlib::rgb_pixel>> test_faces;
    test_faces.push_back(test_face_chip);
    //----using the resnet to extract the feature of face img-----
    std::vector<dlib::matrix<float,0,1>> test_face_descriptor = this->net(test_faces);
    //----create a vector to save relevant feature information----
    std::vector<dlib::matrix<float, 0, 1>> test_vec;
    test_vec.push_back(test_face_descriptor[0]);
        
    return test_vec;
}


//----------------extract face chip in current frame from cam---------------
//----------------将摄像头中的人脸,读取人脸的特征为128维的数据--------------------
/*
    @param:
    1) cv:Mat frame--------------current frame
    2) dlib::shape_discriptor----provide all feature points position for resnet
    3) anet_type net-------------resnet for extracting face feature 
    4) dlib::detector------------face detector
    @return:
    1) std::vector<dlib::matrix<float, 0, 1>>----all faces feature in current frame
*/
std::vector<dlib::matrix<float, 0, 1>> FaceRec::extract_face_feature(std::vector<dlib::rectangle> dets, cv::Mat frame)
{
    dlib::matrix<dlib::bgr_pixel> recog_img = MatToDlib(frame);  
    std::vector<dlib::matrix<float, 0, 1>> vec;
    if(dets.size()>0)
    {
        for(int i=0; i<dets.size(); i++){
            std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
            dlib::matrix<dlib::bgr_pixel> face_chip;
            auto shape = FaceRec::sp(recog_img, dets[i]);
            //从图片中截取人脸图像，并保存为150*150的图片于face_chip中
            dlib::extract_image_chip(recog_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

            cv::Mat tmp_mat =  dlib::toMat(face_chip);//把bgr——pixel的图转为opencvMat
            cv::cvtColor(tmp_mat, tmp_mat, cv::COLOR_BGR2RGB);//转为原色通道为rgb
            dlib::matrix<dlib::rgb_pixel> face_chip_rgb;
            dlib::assign_image(face_chip_rgb, dlib::cv_image<dlib::rgb_pixel>(tmp_mat));
            faces.push_back(std::move(face_chip_rgb)); // std::move将对象标记为可变的右值对象
            std::vector<dlib::matrix<float, 0, 1>> face_descriptor = this->net(faces);//将人脸特征保存为128维的向量特征          
            
            vec.push_back(face_descriptor[0]); //将第一张人脸的128D特征存入vec中
        }
        
    }
    return vec;
}


//------------cal the euclidean dist between any two vector-----------
/*------------------计算两个向量之间的欧式距离----------------------------
    @param:
    1) dlib::matrix<float, 0, 1> db_face_vec----来自于数据库中的一个人脸特征
    2) dlib::matrix<float, 0 ,1> cam_face_vec---来自于摄像头的截取的人脸的特征
    @return:
    1) float error------------------------------两个向量之间的欧式距离,(这里用作误差)
*/
float FaceRec::cal_dist(dlib::matrix<float, 0, 1> db_face_vec, dlib::matrix<float, 0, 1> cam_face_vec)
{
    float error = (double)dlib::length(cam_face_vec- db_face_vec);    
    return error;
}


/*-----------------------从文件的路径中获取文件名------------------------
    @param:
    1) std::string strFullName------------------文件的路径
    @return
    1) std::string -----------------------------文件名
*/
std::string FaceRec::GetPathOrURLShortName(std::string strFullName)
{
    if (strFullName.empty())
    {
        return "";
    }
//    string_replace(strFullName, "/", "\\");
    std::string::size_type iPos = strFullName.find_last_of('/') + 1;
    std::string::size_type iEnd = strFullName.find_last_of('.');
    return strFullName.substr(iPos, iEnd - iPos);
}


/*---------------------为识别到的人脸进行画框--------------------
    @param
    1) cv:Mat             Img -------------------rtsp中的帧图
    2) dlib::rectangle    dets-------------------在帧图中检测到的人脸的位置信息 
    3) const std::string& name-------------------识别成功后,对应人脸的人名
    @return
    1) cv:Mat             Img -------------------返回已经画上框的Img
*/
cv::Mat FaceRec::showImgAndBdBox(cv::Mat Img, dlib::rectangle dets, const std::string& name)
{
        
    int shape_0 = Img.rows;
    int shape_1 = Img.cols;    
    std::vector<int> vec_rec_img;
    vec_rec_img.push_back(dets.top());
    vec_rec_img.push_back(dets.right());
    vec_rec_img.push_back(dets.bottom());
    vec_rec_img.push_back(dets.left());

    //-------------bounding box information face_location------------    
    std::vector<int> bounding_box;
    bounding_box.push_back(MAX(0, vec_rec_img[0]));
    bounding_box.push_back(MIN(shape_1, vec_rec_img[1]));
    bounding_box.push_back(MIN(shape_0, vec_rec_img[2]));
    bounding_box.push_back(MAX(0, vec_rec_img[3]));
    // cv::cvtColor(Img, Img, cv::COLOR_RGB2BGR);
    putText(Img, name, cv::Point(bounding_box[3], bounding_box[0]- 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));            
    cv::rectangle(Img, cv::Point(bounding_box[3], bounding_box[0]), cv::Point(bounding_box[1],bounding_box[2]), cv::Scalar(0,0,255),2,8);

                             
    return Img;        
}