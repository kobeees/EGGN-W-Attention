## Settings

All experiments were trained with a batch size of 32 on an NVIDIA GeForce RTX 3090 GPU. We follow the same split settings with VRCNet to compare our method with other methods fairly. Following the previous method, we adopt $CD_{L2}$ as the main evaluation criterion.

## Prerequisite

```shell
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../../KNN_CUDA-master
python setup.py install
cd ../pointnet2_ops_lib
python setup.py install
```

**Hint**: Don't compile on the Windows platform.

As for other modules, please install by:

```shell
pip install -r requirements.txt
```

### Pretrained Models

We provide our pre-trained models:

|           | url                                                          | performance  |
| --------- | ------------------------------------------------------------ | ------------ |
| num=2048  | [[Google Drive](https://drive.google.com/file/d/1Ke16BbhRpdAMnMi9Tw9PVZw8tDO1Gbbu/view?usp=drive_link)] / [[BaiDuYun](https://pan.baidu.com/s/1Y0wpvVgdGG93I3R_yKy9lA?pwd=njiz )] (code:njiz) | CD = 5.58e-4 |
| num=4096  | [[Google Drive](https://drive.google.com/file/d/1i9Tj0_g9wx-yzn--CisfhPOZG4_jfm64/view?usp=drive_link)] / [[BaiDuYun](https://pan.baidu.com/s/1S0o_56wjU6aCRo_qFYs0FA?pwd=ksla)] (code:ksla ) | CD = 4.18e-4 |
| num=8192  | [[Google Drive](https://drive.google.com/file/d/1Bh7L19ZWcfNusJ-ysC8I2Rd5NwItlyL3/view?usp=drive_link)]  / [[BaiDuYun](https://pan.baidu.com/s/1jRjclghgzzzQhuTbtjYRAw?pwd=jjyu)] (code:jjyu) | CD = 3.32e-4 |
| num=16384 | [[Google Drive](https://drive.google.com/file/d/10mN6yB2cdTPw4l1KzJYrFHY97449tuYb/view?usp=drive_link)]  / [[BaiDuYun](https://pan.baidu.com/s/1o7eFscs3e-kZoKQBbbrUeA?pwd=oel2)] (code:oel2 ) | CD = 2.63e-4 |

### Usage
+ To train a model: run `python train.py -c *.yaml`, e.g. `python train.py -c network.yaml`

+ To test a model: run `python test.py -c *.yaml`, e.g. `python test.py -c network.yaml`

+ Config for each algorithm can be found in `cfgs/`.

   





