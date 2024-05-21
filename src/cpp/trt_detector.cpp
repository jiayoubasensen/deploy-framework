#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "coco_labels.hpp"

using namespace std;
using namespace nvinfer1;

namespace model{

namespace detector {

float iou_calc(bbox bbox1, bbox bbox2){
    auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
    auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
    auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
    auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

    float inter_w = inter_x1 - inter_x0;
    float inter_h = inter_y1 - inter_y0;
    
    float inter_area = inter_w * inter_h;
    float union_area = 
        (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) + 
        (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - 
        inter_area;
    
    return inter_area / union_area;
}


void Detector::setup(void const* data, size_t size) {
   /*
     * detector setup需要做的事情
     *   创建engine, context
     *   设置bindings。这里需要注意，不同版本的yolo的输出binding可能还不一样
     *   分配memory空间。这里需要注意，不同版本的yolo的输出所需要的空间也还不一样
     */

    m_runtime     = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
    m_engine      = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
    m_context     = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
    // m_inputDims   = m_context->getBindingDimensions(0);
    // m_outputDims  = m_context->getBindingDimensions(1);

    m_inputDims   = m_context->getBindingDimensions(0);
    m_c2iDims = m_context->getBindingDimensions(1);
    m_c2i_invDims = m_context->getBindingDimensions(2);
    m_labelDims = m_context->getBindingDimensions(3);
    m_outputDims = m_context->getBindingDimensions(4);
    m_scoreDims = m_context->getBindingDimensions(5);
    m_dirDims = m_context->getBindingDimensions(6);
    m_attrDims = m_context->getBindingDimensions(7);

    CUDA_CHECK(cudaStreamCreate(&m_stream));

    m_inputSize     = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_imgArea       = m_params->img.h * m_params->img.w;
    m_outputSize    = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
    //以下为m_input与output补充数据
    m_c2i_inputSize = m_c2iDims.d[0] * m_c2iDims.d[1] * sizeof(float);
    m_c2i_inv_inputSize =m_c2i_invDims.d[0] * m_c2i_invDims.d[1] * sizeof(float);
    m_output_labelSize = m_labelDims.d[1] * sizeof(float);
    m_output_scoreSize = m_scoreDims.d[1] * sizeof(float);
    m_output_dirSize = m_dirDims.d[1] * sizeof(float);
    m_output_attrSize = m_attrDims.d[1] * sizeof(float);

    // 这里对host和device上的memory一起分配空间
    // CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    // CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
    // CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    // CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[2], m_c2i_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[4], m_c2i_inv_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_output_labelSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[2], m_outputSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[4], m_output_scoreSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[6], m_output_dirSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[8], m_output_attrSize));

    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[3],m_c2i_inputSize));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[5],m_c2i_inv_inputSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_output_labelSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[3], m_outputSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[5], m_output_scoreSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[7], m_output_dirSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[9], m_output_attrSize));

    // 创建m_bindings，之后再寻址就直接从这里找
    // m_bindings[0] = m_inputMemory[1];
    // m_bindings[1] = m_outputMemory[1];
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_inputMemory[3];
    m_bindings[2] = m_inputMemory[5];
    m_bindings[3] = m_outputMemory[1];
    m_bindings[4] = m_outputMemory[3];
    m_bindings[5] = m_outputMemory[5];
    m_bindings[6] = m_outputMemory[7];
    m_bindings[7] = m_outputMemory[9];

}

void Detector::reset_task(){
    m_bboxes.clear();
}

bool Detector::preprocess_cpu() {
    // /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

    // /*Preprocess -- 读取数据*/
    // m_inputImage = cv::imread(m_imagePath);
    // if (m_inputImage.data == nullptr) {
    //     LOGE("ERROR: Image file not founded! Program terminated"); 
    //     return false;
    // }

    // /*Preprocess -- 测速*/
    // m_timer->start_cpu();

    // /*Preprocess -- resize(默认是bilinear interpolation)*/
    // cv::resize(m_inputImage, m_inputImage, 
    //            cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    // int index;
    // int offset_ch0 = m_imgArea * 0;
    // int offset_ch1 = m_imgArea * 1;
    // int offset_ch2 = m_imgArea * 2;
    // for (int i = 0; i < m_inputDims.d[2]; i++) {
    //     for (int j = 0; j < m_inputDims.d[3]; j++) {
    //         index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
    //         m_inputMemory[0][offset_ch2++] = m_inputImage.data[index + 0] / 255.0f;
    //         m_inputMemory[0][offset_ch1++] = m_inputImage.data[index + 1] / 255.0f;
    //         m_inputMemory[0][offset_ch0++] = m_inputImage.data[index + 2] / 255.0f;
    //     }
    // }

    // /*Preprocess -- 将host的数据移动到device上*/
    // CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    // m_timer->stop_cpu();
    // m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");
    // return true;
//--------------------------------------------------------------------------------------------
 
 
 /*fcos3d Preprocess -- 读取数据*/
    m_inputImage = cv::imread(m_imagePath);
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    /*Preprocess -- 测速开始*/
    m_timer->start_cpu();
    // cv::Scalar mean = cv::Scalar(103.530, 116.280, 123.675);
    // cv::Scalar std = cv::Scalar(1,1,1);
    float mean[3] = {103.530, 116.280, 123.675};
    float std[3] = {1,1,1};

    // m_inputImage.convertTo(m_inputImage, CV_32F);

    // m_inputImage -= mean;
    // m_inputImage /=std;
    cv::resize(m_inputImage, m_inputImage, cv::Size(m_params->img.w, m_params->img.h),0,0,cv::INTER_LINEAR);
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            m_inputMemory[0][offset_ch0++] = (m_inputImage.data[index + 0]-mean[0])/std[0];
            m_inputMemory[0][offset_ch1++] = (m_inputImage.data[index + 1]-mean[1])/std[1];
            m_inputMemory[0][offset_ch2++] = (m_inputImage.data[index + 2]-mean[2])/std[2];
        }
    }
    const float c2i_data[9] = {1219.08, 0., 805.8, 0., 1232.41, 478.171, 0., 0., 1};
    const float c2i_inv_data[9] = {8.2029e-04,  0.0000e+00, -6.6099e-01,  0.0000e+00,  8.1142e-04, -3.8800e-01, 0.0000e+00,  0.0000e+00,  1.0000e+00};
    std::memcpy(m_inputMemory[2], c2i_data, m_c2i_inputSize);
    std::memcpy(m_inputMemory[4], c2i_inv_data, m_c2i_inv_inputSize);
    // cv::imwrite("output_1216x1600image.jpg",m_inputImage);
    // std::cout<<"do it finish!"<<std::endl;
    // std::cout << "Image saved successfully." << std::endl;
    // 打印图像的数据
     /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[3], m_inputMemory[2], m_c2i_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[5], m_inputMemory[4], m_c2i_inv_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));
    /*Preprocess -- 测速截止*/
    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("fcos3d_preprocess(CPU)");

    return true;
}


// bool Detector::preprocess_fcos3d_cpu(){

//     /*Preprocess -- 读取数据*/
//     m_inputImage = cv::imread(m_imagePath);
//     if (m_inputImage.data == nullptr) {
//         LOGE("ERROR: Image file not founded! Program terminated"); 
//         return false;
//     }

//     /*Preprocess -- 测速开始*/
//     m_timer->start_cpu();
//     cv::Scalar mean = cv::Scalar(103.530, 116.280, 123.675);
//     cv::Scalar std = cv::Scalar(1,1,1);
//     m_inputImage.convertTo(m_inputImage, CV_32F);

//     m_inputImage -= mean;
//     m_inputImage /=std;

//     cv::resize(m_inputImage, m_inputImage, cv::Size(m_params->img.w, m_params->img.h));
//     m_inputImage = NHWC_to_NCHW(m_inputImage);
//     std::cout << "Channels: " << m_inputImage.channels() << std::endl;
//     std::cout << "Rows: " << m_inputImage.rows << std::endl;
//     std::cout << "Cols: " << m_inputImage.cols << std::endl;

//     // 打印图像的数据
//     std::cout << "Image Data:" << std::endl;
//     std::cout << m_inputImage << std::endl;
//     /*Preprocess -- 测速截止*/
//     m_timer->stop_cpu();
//     m_timer->duration_cpu<timer::Timer::ms>("fcos3d_preprocess(CPU)");
//     return true;
// }


bool Detector::preprocess_gpu() {
    /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

    /*Preprocess -- 读取数据*/
    m_inputImage = cv::imread(m_imagePath);
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }
    
    /*Preprocess -- 测速*/
    m_timer->start_gpu();

    /*Preprocess -- 使用GPU进行warpAffine, 并将结果返回到m_inputMemory中*/
    preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                   m_params->img.h, m_params->img.w, 
                                   preprocess::tactics::GPU_WARP_AFFINE);

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}
std::vector<float> operator*(const std::vector<float>& vec1, const std::vector<float>& vec2) { //运算符重载，实现两个一维vector各个元素相乘
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vector sizes are not equal, element-wise multiplication is not possible.");
    }

    std::vector<float> result;
    result.reserve(vec1.size());

    for (size_t i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] * vec2[i]);
    }

    return result;
}

std::vector<float> einsum(std::vector<float> *corners, std::vector<float> *rot_mat_T){  //torch::einsum("aij,jka->aik")
    int num = (*corners).size();
    std::vector<float> new_corners(num, 0.0f);
    auto a = *rot_mat_T;
    auto b = *corners;
    // std::vector<std::vector<std::vector<float>>> arr(num/24, std::vector<std::vector<float>>(8, std::vector<float>(3, 0.0f)));
    for(int q = 0; q<(num/24); q++){  //a:N
        for(int w = 0; w < 8; w++){ //i:N
            for(int e = 0; e < 3; e++){ //k:3
                for(int r = 0; r < 3; r++){ //j
                    // arr[q][w][e] += (*corners)[q*8*3 + 3*w + r] * (*rot_mat_T)[r*3*(num/24) + e*(num/24) + q];
                    new_corners[q*24+3*w+e] +=(*corners)[q*8*3 + 3*w + r] * (*rot_mat_T)[r*3*(num/24) + e*(num/24) + q];
                }
            }
        }
    }
    return new_corners;
}

bool Detector::postprocess_cpu() {
    // m_timer->start_cpu();

    // /*Postprocess -- 将device上的数据移动到host上*/
    // int output_size = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
    // CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    // CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // /*Postprocess -- yolov8的postprocess需要做的事情*/
    // /*
    //  * 1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
    //  * 2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
    //  * 3. 把最终得到的bbox绘制到原图中
    //  */

    // float conf_threshold = 0.25; //用来过滤decode时的bboxes
    // float nms_threshold  = 0.45;  //用来过滤nms时的bboxes

    // /*Postprocess -- 1. decode*/
    // /*
    //  * 我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
    //  * 几个步骤:
    //  * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height
    //  * 2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
    //  * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
    //  * 4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
    //  * 5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
    //  */
    // int    boxes_count = m_outputDims.d[1];
    // int    class_count = m_outputDims.d[2] - 4;
    // float* tensor;

    // float  cx, cy, w, h, obj, prob, conf;
    // float  x0, y0, x1, y1;
    // int    label;

    // for (int i = 0; i < boxes_count; i ++){
    //     tensor = m_outputMemory[0] + i * m_outputDims.d[2];
    //     label  = max_element(tensor + 4, tensor + 4 + class_count) - (tensor + 4);
    //     conf   = tensor[4 + label];
    //     if (conf < conf_threshold) 
    //         continue;

    //     cx = tensor[0];
    //     cy = tensor[1];
    //     w  = tensor[2];
    //     h  = tensor[3];
        
    //     x0 = cx - w / 2;
    //     y0 = cy - h / 2;
    //     x1 = x0 + w;
    //     y1 = y0 + h;

    //     // 通过warpaffine的逆变换得到yolo feature中的x0, y0, x1, y1在原图上的坐标
    //     preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
    //     preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);
        
    //     bbox yolo_box(x0, y0, x1, y1, conf, label);
    //     m_bboxes.emplace_back(yolo_box);
    // }
    // LOGD("the count of decoded bbox is %d", m_bboxes.size());
    

    // /*Postprocess -- 2. NMS*/
    // /* 
    //  * 几个步骤:
    //  * 1. 做一个IoU计算的lambda函数
    //  * 2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
    //  * 3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
    //  *    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold
    //  */

    // vector<bbox> final_bboxes;
    // final_bboxes.reserve(m_bboxes.size());
    // std::sort(m_bboxes.begin(), m_bboxes.end(), 
    //           [](bbox& box1, bbox& box2){return box1.confidence > box2.confidence;});

    // /*
    //  * nms在网上有很多实现方法，其中有一些是根据nms的值来动态改变final_bboex的大小(resize, erease)
    //  * 这里需要注意的是，频繁的对vector的大小的更改的空间复杂度会比较大，所以尽量不要这么做
    //  * 可以通过给bbox设置skip计算的flg来调整。
    // */
    // for(int i = 0; i < m_bboxes.size(); i ++){
    //     if (m_bboxes[i].flg_remove)
    //         continue;
        
    //     final_bboxes.emplace_back(m_bboxes[i]);
    //     for (int j = i + 1; j < m_bboxes.size(); j ++) {
    //         if (m_bboxes[j].flg_remove)
    //             continue;

    //         if (m_bboxes[i].label == m_bboxes[j].label){
    //             if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
    //                 m_bboxes[j].flg_remove = true;
    //         }
    //     }
    // }
    // LOGD("the count of bbox after NMS is %d", final_bboxes.size());


    // /*Postprocess -- draw_bbox*/
    // /*
    //  * 几个步骤
    //  * 1. 通过label获取name
    //  * 2. 通过label获取color
    //  * 3. cv::rectangle
    //  * 4. cv::putText
    //  */
    // string tag   = "detect-" + getPrec(m_params->prec);
    // m_outputPath = changePath(m_imagePath, "../result", ".png", tag);

    // int   font_face  = 0;
    // float font_scale = 0.001 * MIN(m_inputImage.cols, m_inputImage.rows);
    // int   font_thick = 2;
    // int   baseline;
    // CocoLabels labels;

    // LOG("\tResult:");
    // for (int i = 0; i < final_bboxes.size(); i ++){
    //     auto box = final_bboxes[i];
    //     auto name = labels.coco_get_label(box.label);
    //     auto rec_color = labels.coco_get_color(box.label);
    //     auto txt_color = labels.get_inverse_color(rec_color);
    //     auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
    //     auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thick, &baseline);

    //     int txt_height = txt_size.height + baseline + 10;
    //     int txt_width  = txt_size.width + 3;

    //     cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height - baseline + font_thick)));
    //     cv::Rect  txt_rec(round(box.x0 - font_thick), round(box.y0 - txt_height), txt_width, txt_height);
    //     cv::Rect  box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

    //     cv::rectangle(m_inputImage, box_rec, rec_color, 3);
    //     cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
    //     cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color, font_thick, 16);

    //     LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
    //         name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);

    // }
    // LOG("\tSummary:");
    // LOG("\t\tDetected Objects: %d", final_bboxes.size());
    // LOG("");

    // m_timer->stop_cpu();
    // m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    // cv::imwrite(m_outputPath, m_inputImage);
    // LOG("\tsave image to %s\n", m_outputPath.c_str());

    // // return true;


    //----------------------------以下为fcos3d------------------------------
     m_timer->start_cpu();

    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
    int output_scores_size = m_scoreDims.d[1] * sizeof(float);

    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[2], m_outputMemory[3], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[4], m_outputMemory[5], output_scores_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    //----------------boxTransformer3D--------------------

    std::vector<float> mul_matrix  = {0, 0.5, 0};
    for(int i=0;i<m_outputDims.d[1];i++){
        for(int j=0; j<3; j++){
            m_outputMemory[2][i * m_outputDims.d[2] + j] += m_outputMemory[2][i * m_outputDims.d[2] + j + 3] * mul_matrix[j];
        }
    }       

    std::vector<float> outputs_box;
    for(int i=0;i<m_outputDims.d[1];i++){
        for(int j =0;j<m_outputDims.d[2];j++){
            outputs_box.push_back(m_outputMemory[2][i*m_outputDims.d[2]+j]);
        }
    }
    std::vector<float> outputs_scores;
    for(int i=0; i<m_scoreDims.d[1];i++){
        outputs_scores.push_back(m_outputMemory[4][i]);
    }
    //-----------------show_result_meshlab-----------------
    int score_thr = 0.3;
   int nms_count = std::count_if(outputs_scores.begin(), outputs_scores.end(), [](float value) {
        return value > 0.3;
    });

    outputs_box.resize(9*nms_count);

    //-----------------CamerInstance3DBoxes----------------
    const std::vector<float> corners_norm_ =  //corners_norm原本为3x8维度数据，按一维vector处理
        {-0.5000, -1.0000, -0.5000,
        -0.5000, -1.0000,  0.5000,
        -0.5000,  0.0000,  0.5000,
        -0.5000,  0.0000, -0.5000,
        0.5000, -1.0000, -0.5000,
        0.5000, -1.0000,  0.5000,
        0.5000,  0.0000,  0.5000,
        0.5000,  0.0000, -0.5000};
        std::vector<float> dims;
        std::vector<float> center;
        // std::vector<float> yaw;
        std::vector<float> rot_cos;
        std::vector<float> rot_sin;
        std::vector<float> zeros(nms_count, 0.0f);
        std::vector<float> ones(nms_count, 1.0f);
        std::vector<float> rot_mat_T;
        std::vector<float> corners_norm;

        for(int m = 0; m < nms_count; m++){
            for(int n = 0; n < 8; n++) //dims 与corners_norm相乘广播机制
            {
                dims.push_back(outputs_box[m * 9 + 3]);
                dims.push_back(outputs_box[m * 9 + 4]);
                dims.push_back(outputs_box[m * 9 + 5]);
            }
            center.push_back(outputs_box[m * 9 + 0]);
            center.push_back(outputs_box[m * 9 + 1]);
            center.push_back(outputs_box[m * 9 + 2]);
            rot_cos.push_back(std::cos(outputs_box[ m * 9 + 6]));
            rot_sin.push_back(std::sin(outputs_box[ m * 9 + 6]));
            corners_norm.insert(corners_norm.end(), corners_norm_.begin(), corners_norm_.end());
        }
        rot_mat_T.insert(rot_mat_T.end(), rot_cos.begin(),rot_cos.end());
        rot_mat_T.insert(rot_mat_T.end(), zeros.begin(),zeros.end());

        for(float rot_sin_num: rot_sin){
            rot_mat_T.push_back(- rot_sin_num);
        }

        rot_mat_T.insert(rot_mat_T.end(), zeros.begin(),zeros.end());
        rot_mat_T.insert(rot_mat_T.end(), ones.begin(),ones.end());
        rot_mat_T.insert(rot_mat_T.end(), zeros.begin(),zeros.end());
        rot_mat_T.insert(rot_mat_T.end(), rot_sin.begin(),rot_sin.end());
        rot_mat_T.insert(rot_mat_T.end(), zeros.begin(),zeros.end());
        rot_mat_T.insert(rot_mat_T.end(), rot_cos.begin(),rot_cos.end());
        std::vector<float> corners = dims * corners_norm; //运算符重载实现  
        std::vector<float> new_corners = einsum(&corners, &rot_mat_T);

        for(int n = 0; n < nms_count; n++){
            for(int shape_2 = 0; shape_2 < 8; shape_2++){
                for(int shape_3 = 0; shape_3 < 3; shape_3++)
                    new_corners[n * 24 + shape_2 * 3 + shape_3] += outputs_box[n * 9 + shape_3]; 
            }
        }

        dims.resize(nms_count * 3);
    //-----------------CamerInstance3DBoxes----------------

    std:vector<float> distances;

    for(int i = 0; i < center.size()/3; i++){
        distances.push_back(sqrt(center[i*3] * center[i*3] + center[i*3 + 2] * center[i*3 + 2]));
    }  
    //-----------------points_cam2img----------------------
    std::vector<float> cam2img{1219.08, 0., 805.8, 0., 1232.41, 478.171, 0., 0., 1};
    std::vector<float> cam2img_expand = {1219.08, 0., 805.8, 0., 0., 1232.41, 478.171, 0., 0., 0., 1, 0., 0., 0., 0., 1};
    std::vector<float> points_4d;
    int index = 0;
    const int points_4d_num = new_corners.size()*4/3;

    for(int i =0; i<new_corners.size(); i++){
        if((index +1) %4==0){
            points_4d.push_back(1.0);
            index ++;
        }
        points_4d.push_back(new_corners[i]);
        index ++;
        if(index == points_4d_num -1 ){
            points_4d.push_back(1.0);
        }
    }
    std::vector<float> points_2d(points_4d_num, 0.0f);
    
    for(int row = 0; row<points_4d.size()/4; row++){ //8N
        for(int row_ = 0; row_<cam2img_expand.size()/4; row_++){
            for(int col = 0; col < 4; col++){
                points_2d[row * 4 + row_] += (points_4d[row * 4 + col] * cam2img_expand[row_ * 4 + col]);
            }
        }
    }

    std::vector<float> result_uv;

    for(int i = 0; i < points_2d.size()/4; i++){
        result_uv.push_back(std::round(points_2d[4 * i]/points_2d[4*i + 2] - 1));
        result_uv.push_back(std::round(points_2d[4 * i + 1]/points_2d[4*i + 2] - 1));
    }
    //-----------------points_cam2img----------------------
    
    //-----------------plotRect3DOnImg---------------------
    std::vector<std::pair<int, int>> lineIndices = {{0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {3, 2}, {3, 7},
                                                    {4, 5}, {4, 7}, {2, 6}, {5, 6}, {6, 7}};

    for(int i=0; i<nms_count;i++){
        float distance_ = distances[i];
        for(const auto& indexPair : lineIndices){
            int p1_x = result_uv[16 * i + indexPair.first * 2];
            int p1_y = result_uv[16 * i + indexPair.first * 2 + 1 ];
            int p2_x = result_uv[16 * i + indexPair.second * 2];
            int p2_y = result_uv[16 * i + indexPair.second * 2 + 1];
            cv::Point point1(p1_x, p1_y);
            cv::Point point2(p2_x, p2_y);
            cv::line(m_inputImage, point1, point2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    int pos_x = result_uv[16 * i];
    int pos_y = result_uv[16 * i + 1];

    cv::Point point_pos(pos_x, pos_y);

    cv::putText(m_inputImage, std::to_string(distance_), point_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    //-----------------plotRect3DOnImg---------------------
    //-----------------show_result_meshlab-----------------

    std::cout<<"done!"<<std::endl;
    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("fcos3d_postprocess(CPU)");
    cv::imshow("img_vis", m_inputImage);
    cv::waitKey();

    return true;
}
}

bool Detector::postprocess_gpu() {
    return postprocess_cpu();
}

shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

}; // namespace detector
}; // namespace model
