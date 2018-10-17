#include "spdhelper.hpp"
#include "CaffePredict.h"
#include <string>
#include <vector>
#include <memory>
#include "cmdline.hpp"

void PredictFile(std::shared_ptr<CaffePredict> p, std::string param)
{

}

void PredictDir(std::shared_ptr<CaffePredict> p, std::string param)
{

}

void PredictCap(std::shared_ptr<CaffePredict> p, std::string param)
{

}


int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    

//    if(argc < 3)
//    {
//        std::cout << "Usage: " << argv[0] <<
//        " (file|dir|cap) param"  << std::endl;
//        std::cout << "for file: param is file full path" << std::endl;
//        std::cout << "for dir: param is dir name" << std::endl;
//        std::cout << "for cap: param is cap id" << std::endl;
//        return 0;
//    }
//    std::string model_path = R"()";
//    std::string train_path = R"()";
//    bool use_gpu = true;
//
//    cv::Scalar mean = {104, 117, 123};
//    float scalor = 1.0f;
//    const int INPUT_IMAGE_WIDTH = 320;
//    const int INPUT_IMAGE_HEIGHT = 320;
//    auto p = std::make_shared<CaffePredict>(model_path, train_path, use_gpu);
//    p->SetTransformParam(mean, scalor);
//
//    std::string mode = argv[1];
//    std::string param = argv[2];
//    if (mode == "file")
//    {
//        PredictFile(p, param);
//    }
//    else if(mode == "dir")
//    {
//        PredictDir(p, param);
//    }
//    else
//    {
//        PredictCap(p, param);
//    }
    return 0;
}