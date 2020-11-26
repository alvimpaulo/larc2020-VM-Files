#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"

#ifndef _MODEL_LOADER_HPP_
#define _MODEL_LOADER_HPP_

using namespace std;
using namespace tensorflow;

struct Prediction{
	unique_ptr<vector<vector<float>>> boxes;
	unique_ptr<vector<float>> scores;
	unique_ptr<vector<int>> labels;
};

class ModelLoader
{
	private:
		SavedModelBundle bundle;
		SessionOptions session_options;
		RunOptions run_options;
		void make_prediction(vector<Tensor> &image_output, Prediction &pred);
	public:
		ModelLoader(string);
		void predict(string filename, Prediction &out_pred);
};

#endif
