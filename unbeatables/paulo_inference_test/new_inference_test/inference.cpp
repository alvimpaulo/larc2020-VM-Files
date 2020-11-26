#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "model_loader.hpp"
#include <iostream>
#include <arpa/inet.h>
#include <dirent.h>
#include <stdio.h>
#include <thread>

using namespace tensorflow;
using namespace std::chrono;

struct Result
{
    std::string image_name;
    float score;
    int xmin, ymin, xmax, ymax;
};

const std::string PATH_TO_SAVED_MODEL = "/home/ubuntu/unbeatables/dataset/LARC2020/exported/my_mobilenet3_retrain/saved_model";
const std::string IMAGE_PATH = "/home/ubuntu/test_images/";

template <typename T>
std::vector<std::vector<T>> SplitVector(const std::vector<T> &vec, size_t n)
{
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}

void inference(std::vector<std::string> &image_names, ModelLoader &model)
{

    Prediction out_pred;
    out_pred.boxes = unique_ptr<vector<vector<float>>>(new vector<vector<float>>());
    out_pred.scores = unique_ptr<vector<float>>(new vector<float>());
    out_pred.labels = unique_ptr<vector<int>>(new vector<int>());
    unsigned int width, height;
    Result result;

    for (int i = 0; i < image_names.size(); ++i)
    // for (int i = 0; i < 2000; ++i)
    {
        cv::Mat opencv_img;
        // std::cout << image_names[i] << std::endl;
        opencv_img = cv::imread(image_names[i]);
        int width = opencv_img.cols;
        int height = opencv_img.rows;
        
        ofstream file;
        file.open("/home/ubuntu/detection-results/" + token + ".txt", ios::app);
        file.close();

        // std::cout << image_names[i] << std::endl;

        // std::cout << "w: " << width << "h: " << height << std::endl;
        model.predict(image_names[i], out_pred);
        for (auto &score : (*out_pred.scores))
        {
            if (score < 0.45)
            {
                continue;
            }
            size_t pos = &score - &(*out_pred.scores)[0];

            auto box = (*out_pred.boxes)[pos];
            result.ymin = (int)(box[0] * height);
            result.xmin = (int)(box[1] * width);
            result.ymax = (int)(box[2] * height);
            result.xmax = (int)(box[3] * width);

            char delimiter = '/';
            char delimiter2 = '.';
            size_t pos1 = image_names[i].rfind(delimiter) + 1;
            size_t pos2 = image_names[i].rfind(delimiter2) - pos1;
            std::string token = image_names[i].substr(pos1, pos2);
            ofstream myfile;

            myfile.open("/home/ubuntu/inference_cpp/" + token + ".txt", ios::app);
            myfile << "robot"
                   << " " << score << " " << result.xmin << " " << result.ymin << " " << result.xmax << " " << result.ymax << "\n";
            myfile.close();

            if (i < 20)
            {
                //escreve box nas primeiras 20 imgs
                // and its top left corner...
                cv::Point pt1(result.xmin, result.ymin);
                // and its bottom right corner.
                cv::Point pt2(result.xmax, result.ymax);
                // These two calls...
                cv::rectangle(opencv_img, pt1, pt2, cv::Scalar(0, 0, 255));
            }
        }

        if (i < 20)
        {
            char delimiter = '/';
            char delimiter2 = '.';
            size_t pos1 = image_names[i].rfind(delimiter) + 1;
            size_t pos2 = image_names[i].rfind(delimiter2) - pos1;
            std::string token = image_names[i].substr(pos1, pos2);
            cv::imwrite("/home/ubuntu/inference_cpp_imgs/" + token + ".png", opencv_img);
        }
    }
}

int main(int argc, char **argv)
{

    std::vector<std::string> image_path;
    std::vector<Result> results;
    steady_clock::time_point ini, fim;

    std::cout << "Loading model...";

    ModelLoader model(PATH_TO_SAVED_MODEL);
    std::cout << "Model loaded...";

    DIR *d;
    struct dirent *dir;
    d = opendir("/home/ubuntu/test_images/");
    std::string str;
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            str = dir->d_name;
            str = IMAGE_PATH + str;
            // std::cout << str << std::endl;
            if (str.compare("/home/ubuntu/test_images/.") != 0 && str.compare("/home/ubuntu/test_images/..") != 0)
            {
                image_path.push_back(str);
            }
        }
        closedir(d);
    }

    std::vector<std::vector<std::string>> split_vector = SplitVector(image_path, 3);

    std::vector<std::string> split_lo = split_vector[0];
    std::vector<std::string> split_mid = split_vector[1];
    std::vector<std::string> split_hi = split_vector[2];

    ini = steady_clock::now();
    std::cout << "Tempo de Inicio:" << ini.time_since_epoch().count() << std::endl;
    // std::cout << split_lo.size() << " " << split_mid.size() << " "<< split_hi.size() << " " << std::endl;

    std::thread thread_lo(inference, std::ref(split_lo), std::ref(model));
    std::thread thread_mid(inference, std::ref(split_mid), std::ref(model));
    std::thread thread_hi(inference, std::ref(split_hi), std::ref(model));

    thread_lo.join();
    thread_mid.join();
    thread_hi.join();

    fim = steady_clock::now();

    std::cout << "Finalizado no tempo: " << fim.time_since_epoch().count() << std::endl;
    duration<double> time_span = duration_cast<duration<double>>(fim - ini);
    cout << " - Finalizado em: " << time_span.count() << "s." << endl;
}

// tempo de inferencia: ~17 ms por img