#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "model_loader.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include <arpa/inet.h>

namespace fs = std::filesystem;
using namespace tensorflow;
using namespace std;
using namespace std::chrono;

struct Result{
    std::string image_name;
    float score;
    int xmin, ymin, xmax, ymax;
};

const string PATH_TO_SAVED_MODEL = "/home/ubuntu/unbeatables/dataset/LARC2020/exported/my_mobilenet_best/saved_model";
const string PATH_TO_INFERENCE_FILES = "mAP/input/detection-results/";
const string IMAGE_PATH = "/home/ubuntu/unbeatables/dataset/LARC2020/dataset/";

void inference(std::vector<std::string> &image_names, ModelLoader &model, std::vector<Result> &output){
    Prediction out_pred;
    out_pred.boxes = unique_ptr<vector<vector<float>>>(new vector<vector<float>>());
	out_pred.scores = unique_ptr<vector<float>>(new vector<float>());
	out_pred.labels = unique_ptr<vector<int>>(new vector<int>());
    unsigned int width, height;
    Result result;

    for(int i=0; i<image_names.size(); ++i){
        // std::ifstream in(image_names[i]);

    //     // in.seekg(16);
    //     // in.read((char *)&width, 4);
    //     // in.read((char *)&height, 4);

    //     // width = ntohl(width);
    //     // height = ntohl(height);

    //     width = 640;
    //     height = 480;
        std::cout << image_names[i] <<std::endl;
        model.predict(image_names[i], out_pred);
    //     // for(auto& score: (*out_pred.scores)) {
	// 	// 	if(score < 0.3) {
	// 	// 		continue;
	// 	// 	}
	// 	// 	size_t pos = &score - &(*out_pred.scores)[0];
            
	// 	// 	auto box = (*out_pred.boxes)[pos];
	// 	// 	result.ymin = (int) (box[0] * height);
	// 	// 	result.xmin = (int) (box[1] * width);
	// 	// 	result.ymax = (int) (box[2] * height);
	// 	// 	result.xmax = (int) (box[3] * width);
    //     //     cout << image_names[i];
    //     // }
    }

}


int main(int argc, char **argv){

    std::vector<std::string> image_path;
    std::vector<Result> results;

    std::cout << "Loading model...";
    std::cout << PATH_TO_SAVED_MODEL <<std::endl;
	ModelLoader model(PATH_TO_SAVED_MODEL);
    std::cout << "Model loaded...";

    std::string str;
    for (const auto & entry : fs::directory_iterator(IMAGE_PATH)){

        image_path.push_back((entry.path().string()));       
    }


    inference(image_path,model,results);

    

    

}


