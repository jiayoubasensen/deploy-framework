#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{

    float* test_number = new float[3];
    test_number[0] = 0.0;
    test_number[1] = 0.1;
    test_number[2] = 0.2;

    /*这么实现目的在于让调用的整个过程精简化*/
    // string onnxPath    = "models/onnx/end2end_106m.onnx";
    string onnxPath    = "models/onnx/end2end.onnx";

    auto level         = logger::Level::VERB;
    auto params        = model::Params();
    params.num_cls = 1;
    // params.img         = {640, 640, 3};
    params.img         = {1216, 1600, 3};  //HWc
    params.task        = model::task_type::DETECTION;
    params.dev         = model::device::CPU;
    params.prec        = model::precision::FP16;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, level, params);
    // 根据worker中的task类型进行推理
\
    worker->inference("/home/sen/projects/bushu/deploy-fcos3d/output_1216x1600image.jpg");
    worker->inference("/home/sen/projects/bushu/deploy-fcos3d/output_image.jpg");
    worker->inference("/home/sen/projects/bushu/deploy-fcos3d/output_1216x1600image.jpg");
    worker->inference("/home/sen/projects/bushu/deploy-fcos3d/output_image.jpg");
    // worker->inference("data/source/crossroad.jpg");
    // worker->inference("data/source/airport.jpg");
    // worker->inference("data/source/bedroom.jpg");
    return 0;
}

