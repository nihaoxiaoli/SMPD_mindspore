# SMPD-mindspore
This is an implementation of "Stabilizing Multispectral Pedestrian Detection with Evidential Hybrid Fusion" based on MindSpore. 

This repository aims to provide a comprehensive detection process based on MindSpore for multispectral pedestrian detection, making it the first of its kind. We encourage everyone to collaborate and contribute towards the continuous improvement and maintenance of this code.

# Demo
## Prerequisites
You can execute this command to create a conda environment for this repository.

```
conda env create -f smpd_mindspore.yaml
```

Note: Please go to the official website [[link]](https://www.mindspore.cn/install) to install MindSpore for use on Ascend or GPU. 

## Run

1. You can directly download the model parameters at [smpd_mindspore.ckpt](https://pan.baidu.com/s/1llKoQ7U8PVrv2taO-v72jw?pwd=yywb).

2. Add the path of model parameters to the variable ```load_name``` in the file ```detector.py```.

3. Download data Kaist, and add the path of data to the variable ```self._pic_path``` in the file ```kaist.py```.

4. To test on kaist, simply run:

    ```
    python detector.py --dataset kaist --net vgg16 --reasonable
    ```

Note: Due to different implementations of RoIPooling operations, the results may differ slightly from those based on other frameworks. We have tested and found that the results of other operations are strictly consistent.