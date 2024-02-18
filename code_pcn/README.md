## Settings

All experiments were trained with a batch size of 32 on an NVIDIA GeForce RTX 3090 GPU. The initial learning rate is set to 0.001 with a continuous decay of 0.5 for every 50 epochs. We follow the same split settings with PCN to compare our method with other methods fairly. Following the previous method, we adopt $CD_{L1}$ as the main evaluation criterion.

## Prerequisite

```shell
cd extensions/chamfer_distance
python setup.py install
cd ../KNN_CUDA-master
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

|                     | url                                                          | performance  |
| ------------------- | ------------------------------------------------------------ | ------------ |
| Our                 | [[Google Drive](https://drive.google.com/file/d/1dXb68kOhkrYRJcs-r8deb8T3AijzhXZN/view?usp=drive_link)] / [[BaiDuYun](https://pan.baidu.com/s/1sEYFRo5FOHUi3ij53GKopA?pwd=2pyz )] (code:2pyz) | CD = 7.01e-3 |
| Our_with_seedformer | [[Google Drive](https://drive.google.com/file/d/1iENRIZzh7vO7HYeSFtTx7_pvG38J8y1x/view?usp=drive_link)] / [[BaiDuYun](https://pan.baidu.com/s/1Hkex1_LNHYUN5iQZdvMMuA?pwd=th39 )] (code:th39) | CD = 6.65e-3 |

## Training

In order to train the model, please use script:

```shell
# For PCN dataset
python train.py

# For KITTI dataset
python train.py --category=='car'
```

## Testing

In order to test the model, please use follow script:

```shell
# For PCN dataset
python test.py

# For KITTI dataset
python test.py --category=='car'
```

The parameter `--novel` is for novel testing data contains unseen categories while training. The parameter `--save` is used for saving the prediction into `.ply` file and visualize the result into `.png` image.



