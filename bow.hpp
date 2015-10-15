#ifndef BOW_H
#define BOW_H

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"

namespace U {

	template <typename T>
	class BOW {
	public:
		BOW(
			cv::Ptr<cv::FeatureDetector> f = new cv::SiftFeatureDetector,
			cv::Ptr<cv::DescriptorExtractor> d = new cv::SiftDescriptorExtractor,
			cv::Ptr<cv::DescriptorMatcher> m = new cv::BFMatcher
		) : _featDetector(f), _descExtractor(d), _descMatcher(m), 
			_bowExtractor(new cv::BOWImgDescriptorExtractor(_descExtractor, _descMatcher)), 
			_classifier(new T) 
		{}

		void Create(const std::vector<cv::Mat>&, int clusterCount = 100);
		
		void Train(const std::vector<cv::Mat>&, const std::vector<int>&);

		double Predict(const cv::Mat&);

		void SaveClassifier(const std::string& file);
		void LoadClassifier(const std::string& file);

	private:
		cv::Ptr<cv::FeatureDetector> _featDetector;
		cv::Ptr<cv::DescriptorExtractor> _descExtractor;
		cv::Ptr<cv::DescriptorMatcher> _descMatcher;

		cv::Ptr<cv::BOWImgDescriptorExtractor> _bowExtractor;

		cv::Ptr<T> _classifier;
	};

	template <typename T>
	void BOW<T>::Create(const std::vector<cv::Mat>& data, int clusterCount) {
		TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON;
		BOWKMeansTrainer bowTrainer(clusterCount, terminate_criterion, 3, KMEANS_PP_CENTERS);

		Mat training_descriptors(0, _descExtractor->descriptorSize(), _descExtractor->descriptorType());

		for (const auto& im : data) {
			std::vector<KeyPoint> keypoints;
			_featDetector->detect(im, keypoints);
			Mat imageDescriptors;
			_descExtractor->compute(im, keypoints, imageDescriptors);
			if (!imageDescriptors.empty()) {
				training_descriptors.push_back(imageDescriptors);
			}
		}
		bowTrainer.add(training_descriptors);

		Mat vocabulary = bowTrainer.cluster();
		_bowExtractor->setVocabulary(vocabulary);
	}

	template <typename T>
	void BOW<T>::Train(const std::vector<cv::Mat>& data, const std::vector<int>& labels) {
		if (data.size() != labels.size()) {
			throw std::runtime_error("All inputs must have a label.");
		}

		Mat trainData(0, _bowExtractor->getVocabulary().rows, CV_32FC1);
		Mat classes(0, 1, CV_32SC1);
		int i = 0;
		for (const auto& im : data) {
			std::vector<KeyPoint> keypoints;
			_featDetector->detect(im, keypoints);
			Mat imageDescriptors;
			_descExtractor->compute(im, keypoints, imageDescriptors);
			if (!imageDescriptors.empty()) {
				Mat response_hist;
				_bowExtractor->compute(im, keypoints, response_hist);
				trainData.push_back(response_hist);
				classes.push_back(lab.at(i));
			}
			i++;
		}

		classifier_traits<T>::Train(*_classifier, trainData, classes);
	}

	template <typename T>
	double BOW<T>::Predict(const cv::Mat& data) {
		std::vector<KeyPoint> keypoints;

		_featDetector->detect(data, keypoints);

		Mat respHist;
		_bowExtractor->compute(data, keypoints, respHist);
		return _classifier->predict(respHist);
	}

	template <typename T>
	void BOW<T>::SaveClassifier(const std::string& file) {
		_classifier->save(file.c_str());
	}

	template <typename T>
	void BOW<T>::LoadClassifier(const std::string& file) {
		_classifier->load(file.c_str());
	}

	template <typename T>
	struct classifier_traits {
		static void Train(T&, const cv::Mat&, const cv::Mat&);// { throw std::runtime_error("Unknown classifier strategy."); }
	};

	template <>
	static void classifier_traits<cv::RandomTrees>::Train(cv::RandomTrees& t, const cv::Mat& trainData, const cv::Mat& classes) {
		t.train(trainData, CV_ROW_SAMPLE, classes);
	}

	template <>
	static void classifier_traits<cv::Boost>::Train(cv::Boost& t, const cv::Mat& trainData, const cv::Mat& classes) {
		t.train(trainData, CV_ROW_SAMPLE, classes);
	}


	template <>
	static void classifier_traits<cv::SVM>::Train(cv::SVM& t, const cv::Mat& trainData, const cv::Mat& classes) {
		t.train(trainData, classes);
	}


	template <>
	static void classifier_traits<cv::NormalBayesClassifier>::Train(cv::NormalBayesClassifier& t, const cv::Mat& trainData, const cv::Mat& classes) {
		t.train(trainData, classes);
	}

}

#endif
