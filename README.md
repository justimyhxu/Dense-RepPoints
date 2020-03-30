# Dense Reppoints: Representing Visual Objects with Dense Point Sets

The major contributors include [Yinghao Xu*](https://github.com/justimyhxu), [Ze Yang*](https://yangze.tech/), [Han Xue*](https://github.com/xiaoxiaoxh), [Zheng Zhang](https://www.microsoft.com/en-us/research/people/zhez/) ,[Han Hu](https://ancientmooner.github.io/). (* indicates equal contribution)

This repo is a official implementation of "Dense RepPoints: Representing Visual Objects with Dense Point Sets" on COCO object detection based on open-mmlab's mmdetection. Many thanks to mmdetection for their simple and clean framework.


## Introduction
We present a new object representation, called Dense Rep-Points, which utilize a large number of points to describe the multi-grained object representation of both box level and pixel level. 
Techniques are proposed to efficiently  process  these  dense  points, which  maintains  nearconstant complexity with increasing point number. 
The dense RepPointsis  proved  to  represent  and  learn  object  segment  well, by  a  novel  distance transform sampling method combined with a set-to-set supervision.
The novel distance transform sampling method combines the strength of contour and grid representation, which significantly outperforms the counter-parts using contour or grid representations. 
The Dense RepPoints proves to perform significantly better than previous approaches or act as the ﬁrst effort in representing several irregular fine-grained object shapes, substantially enlarging the research breadth for object shape analysis.
<div align="center">
  <img src="demo/dense_reppoints.png" width="300px" />   <img src="demo/inference_mask.png" width="300px" />
  <p>Learning Dense RepPoints in Object Detection and Instance Segmentation.</p>
</div>

## Usage

a. Clone the repo, install mmdetection and download the COCO detection dataset.
```
git clone --recursive https://github.com/justimyhxu/Dense-RepPoints.git
```
Please refer to mmdetection [INSTALL.md](./docs/INSTALL.md) for installation and dataset preparation in details.

b. Run experiments with a specific configuration file:
```
./tools/dist_train.sh ${path-to-cfg-file} ${num_gpu} --validate
```
We give one example here:
```
./tools/dist_train.sh configs/dense_reppoints/dense_reppoints_729pts_r50_fpn_1x.py 8 --validate
```

d. For evaluation, output file is required
```
./tools/dist_test.sh  ${path-to-cfg-file}  ${model_path} ${num_gpu} --out ${out_file}
```
Please see [GETTING_STARTED.md](./docs/GETTING_STARTED.md) for the basic usage of MMDetection in details

## Citing Dense RepPoints

```
@article{yang2019dense,
  title={Dense reppoints: Representing visual objects with dense point sets},
  author={Yang, Ze and Xu, Yinghao and Xue, Han and Zhang, Zheng and Urtasun, Raquel and Wang, Liwei and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:1912.11473},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the table below.


| Method          | Backbone | Anchor    |  convert func | Refine Assigner  | Lr schd | box AP   | mask AP    | Download |
| :----:          | :------: | :-------: | :------:      | :-----:| :-----: | :----:  | :------: | :------:  |
| Dense RepPoints | R-50-FPN | none      | MinMax        | MaxIou           |1x       | 38.9     | 32.7     | [model](https://drive.google.com/file/d/1gAgGTqsrufRleYflrClxI0AUuUC_x1MN/view?usp=sharing) |
| Dense RepPoints | R-50-FPN | none      | MinMax        | Atss             |1x       | 39.5     | 32.9     | [model](https://drive.google.com/file/d/1jhojzH0N9KI2SpLa-xNQZHZUCU9O5SAE/view?usp=sharing) |
| Dense RepPoints | R-50-FPN | none      | MinMax        | Atss             |3x       | 43.0     | 36.2     | [model](https://drive.google.com/file/d/1ZPd1iCZGEzqhVs4PwCXdThzqp0ufa_KX/view?usp=sharing) |


**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively. 
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `none` in the `anchor` column means 2-d `center point` (x,y) is used to represent the initial object hypothesis. 
- `ms` denotes multi-scale training or multi-scale test.


