#include "CaffePredict.h"


int CaffePredict::GetNumOutput()
{
	if (net_)
		return net_->num_outputs();
	return 0;
}

CaffePredict::~CaffePredict()
{
}


CaffePredict::CaffePredict(const string& model_file, const string& trained_file, bool use_gpu)
{
	if (use_gpu)
		Caffe::set_mode(Caffe::GPU);
	else
		Caffe::set_mode(Caffe::CPU);

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	/* init variable input_geometry*/
	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	/*CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";*/
	//  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	//init mean and sclor
	globalMean_ = cv::Scalar{0,0,0};
	scale_ = 1.0f;
}

void CaffePredict::SetTransformParam(const string & mean_file, float scale)
{
	scale_ = scale;
	SetMean(mean_file);
}

void CaffePredict::SetTransformParam(cv::Scalar mean, float scale)
{
	scale_ = scale;
	globalMean_ = mean;
}


/* Load the mean file in binaryproto format. */
void CaffePredict::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	globalMean_ = cv::mean(mean);
	//cv::Scalar channel_mean = cv::mean(mean);
	//mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<std::vector<float>> CaffePredict::Predict(const cv::Mat& img)
{
	//1.
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	//2. pre-process
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	Preprocess(img, &input_channels);

	/*forward*/
	net_->Forward();

	/* Copy the output layer to a std::vector */
	auto num_output = net_->num_outputs();
	std::vector<std::vector<float>> out;
	for (int i = 0; i < num_output; i++)
	{
		auto* output_layer = net_->output_blobs()[i];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->count();
		out.push_back(std::vector<float>(begin, end));
	}
	return out;
}
std::vector<std::pair<std::vector<int>, std::vector<float>>> CaffePredict::Predict(const cv::Mat& img, const std::vector<std::string>& blob_names)
//std::vector<float> CaffePredict::Predict(const cv::Mat & img, std::string blob)
{
	//1.
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	//2. pre-process
	//BTimer t;
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	Preprocess(img, &input_channels);
	//LOGI("inner preprocess time cost: {}", t.elapsed());

	/*forward*/
	//t.reset();
	net_->Forward();
	//LOGI("inner forward time cost: {}", t.elapsed());

	/*copy the spec layer*/
	std::vector<std::pair<std::vector<int>, std::vector<float>>> res;
	for (auto& name : blob_names)
	{
		std::pair<std::vector<int>, std::vector<float>> out;
		auto outblob = net_->blob_by_name(name);
		const float* begin = outblob->cpu_data();
		const float* end = begin + outblob->count();
		out.second = std::move(std::vector<float>(begin, end));
		out.first = outblob->shape();
		res.emplace_back(out);
	}
	return res;
}

std::vector<std::pair<std::vector<int>, std::vector<float>>> CaffePredict::Predict(const std::vector<cv::Mat>& imgs, const std::vector<std::string>& blob_names)
{
	//0.
	std::vector<std::pair<std::vector<int>, std::vector<float>>> res;
	int N = imgs.size();
	if (N <= 0)
		return res;
	int W = imgs[0].cols;
	int H = imgs[0].rows;
	int C = imgs[0].channels();

	//1.
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(N, C, H, W);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	//2. pre-process
	//BTimer t;
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	//Preprocess(img, &input_channels);
	std::vector<cv::Mat> cm;
	cm.resize(C);
	for (int i = 0; i < N; i++)
	{
		cm[0] = std::move(input_channels[i * 3]);
		cm[1] = std::move(input_channels[i * 3 + 1]);
		cm[2] = std::move(input_channels[i * 3 + 2]);
		Preprocess(imgs[i], &cm);
	}
	//LOGI("inner preprocess time: {}", t.elapsed());

	/*forward*/
	//t.reset();
	net_->Forward();
	//LOGI("inner forward time: {}", t.elapsed());

	/*copy the spec layer*/
	
	for (auto& name : blob_names)
	{
		std::pair<std::vector<int>, std::vector<float>> out;
		auto outblob = net_->blob_by_name(name);
		const float* begin = outblob->cpu_data();
		const float* end = begin + outblob->count();
		out.second = std::move(std::vector<float>(begin, end));
		out.first = outblob->shape();
		res.emplace_back(out);
	}
	return res;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void CaffePredict::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int N = input_layer->shape()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int n = 0; n < N; n++)
	{
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}
	
}

void CaffePredict::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized = sample;
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	/* substract mean and scale */
	cv::Mat sample_normalized;
	int type = CV_32FC1;
	if (num_channels_ == 3)
		type = CV_32FC3;
	sample_normalized = sample_float - globalMean_;
	if (std::abs(scale_ - 1.0f) > 0.00001)
		sample_normalized *= scale_;

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	//CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	//	== net_->input_blobs()[0]->cpu_data())
	//	<< "Input channels are not wrapping the input layer of the network.";
}

//spatial softmax
int SpatialSoftmax(std::vector<float>& data, std::vector<float>& shape)
{
	int N, C, H, W;
	if (shape.size() == 4)
	{
		N = shape[0];
		C = shape[1];
		H = shape[2];
		W = shape[3];
	}
	else if (shape.size() == 3)
	{
		N = 1;
		C = shape[0];
		H = shape[1];
		W = shape[2];
	}
	//
	return 0;
}