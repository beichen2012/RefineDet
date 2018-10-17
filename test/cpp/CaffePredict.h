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
	/*��ʼ��������ṹ�ļ���ѵ�������ļ�*/
	CaffePredict(const string& model_file,
		const string& trained_file,
		bool use_gpu = true);

	/*��ֵ������*/
	void SetTransformParam(const string& mean_file, float scale = 1.0f);
	void SetTransformParam(cv::Scalar mean, float scale = 1.0f);

	/*Ԥ�⣬�������е��������*/
	std::vector<std::vector<float>> Predict(const cv::Mat& img);

	/*Ԥ��,����ָ����Ľ��*/
	std::vector<std::pair<std::vector<int>, std::vector<float>>> Predict(const cv::Mat& img, const std::vector<std::string>& blob_names);



	/*������Ԥ��,����ָ����Ľ��*/
	std::vector<std::pair<std::vector<int>, std::vector<float>>> Predict(const std::vector<cv::Mat>& imgs, const std::vector<std::string>& blob_names);

	/*�������ĳ���*/
	int GetNumOutput();

	virtual ~CaffePredict();
private:
	void SetMean(const string& mean_file);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

	/*������Ԥ��*/
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