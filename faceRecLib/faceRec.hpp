
#ifndef FACEREC_H
#define FACEREC_H
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_transforms.h>
using namespace dlib;
class FaceRec{
public:
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
    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
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



    dlib::matrix<dlib::bgr_pixel> MatToDlib(cv::Mat cv_Img);
    cv::Mat showImgAndBdBox(cv::Mat Img, dlib::rectangle dets, const std::string& name);
    std::vector<dlib::matrix<float, 0, 1>> extract_face_feature(const std::string& filePath);        
    std::vector<dlib::matrix<float, 0, 1>> extract_face_feature(std::vector<dlib::rectangle> dets, cv::Mat frame); 
    float cal_dist(dlib::matrix<float, 0, 1> db_face_vec, dlib::matrix<float, 0, 1> cam_face_vec);
    std::string GetPathOrURLShortName(std::string strFullName);
    std::string resnet_model_fileName;
    std::string face_shape_discriptor_fileName;        
    dlib::frontal_face_detector detetor;

    dlib::shape_predictor sp;
    anet_type net;
    void init_();
    // static cv::String rtsp;
    
    
};


#endif