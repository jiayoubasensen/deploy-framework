## 与 mmdeploy 框架不同，我们完全在 C++ 中部署 Pytorch 模型。

## 基于pytorch的实现在C++实现后处理

### Postprocess -- yolov8的postprocess需要做的事情
1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
3. 把最终得到的bbox绘制到原图中

### Postprocess -- 1. decode
我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
1. 从每一个bbox中对应的ch中获取cx, cy, width, height
2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
3. 将cx, cy, width, height转换为x0, y0, x1, y1
4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
    

### Postprocess -- 2. NMS
1. 做一个IoU计算的lambda函数
2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold

### Postprocess -- 3. draw_bbox
1. 通过label获取name
2. 通过label获取color
3. cv::rectangle
4. cv::putText

理解了算法实现起来就很容易了。详细请看代码。

## int8 Calibration
基于classification章节实现的int8 calibrator,我们也可以用在yolov8上。
代码中采用的calibration数据及是coco2017,大家可以根据自己的情况修改。
然而，当我们在做int8推理的时候，我们会发现精度下降非常严重。主要是出现在无法检测到物体。
有关这个问题会在``7.4-quantization-analysis``讲解。
