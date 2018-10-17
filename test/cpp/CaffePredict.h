#pragma once
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
using std::string;
using namespace caffe;

//spatial softmax
int SpatialSoftmax(std::vector<float>& data, std::vector<float>& shape);

class CaffePredict
{
public:
	/*初始化，网络结构文件及训练参数文件*/
	CaffePredict(const string& model_file,
		const string& trained_file,
		bool use_gpu = true);

	/*均值与缩放*/
	void SetTransformParam(const string& mean_file, float scale = 1.0f);
	void SetTransformParam(cv::Scalar mean, float scale = 1.0f);

	/*预测，返回所有的输出层结果*/
	std::vector<std::vector<float>> Predict(const cv::Mat& img);

	/*预测,返回指定层的结果*/
	std::vector<std::pair<std::vector<int>, std::vector<float>>> Predict(const cv::Mat& img, const std::vector<std::string>& blob_names);



	/*按批次预测,返回指定层的结果*/
	std::vector<std::pair<std::vector<int>, std::vector<float>>> Predict(const std::vector<cv::Mat>& imgs, const std::vector<std::string>& blob_names);

	/*输出结果的长度*/
	int GetNumOutput();

	virtual ~CaffePredict();
private:
	void SetMean(const string& mean_file);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

	/*按批次预测*/
	void Preprocess(const std::vector<cv::Mat>& imgs,
		std::vector<cv::Mat> input_channels);

private:
	std::shared_ptr<Net<float> > net_;
	std::vector<string> labels_;
	int num_channels_;

	//mean and scale
	cv::Scalar globalMean_;
	float scale_;
};