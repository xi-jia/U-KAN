
# U-KAN: U-Shape Kolmogorov-Arnold Networks for Image Registration

## A KAN version of U-Net for Image Registration

Key facts of U-KAN:

* The [KAN](https://github.com/KindXiaoming/pykan) parts of the code are adopted from [KA-Conv](https://github.com/XiangboGaoBarry/KA-Conv) and [Fast-KAN](https://github.com/ZiyaoLi/fast-kan).
* The registration parts of the code are adopted from [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration). 
* The data used is from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), where we used the mid-ventricular slices from 100 labeled paired ES and ED scans. Specifically, for each pathology class,  we split them as 4:1:5 for training:validation:testing, resulting in 40 pairs for training, 10 pairs for validatoin, and 50 pairs for testing.

Take home messages from U-KAN:

* The registration performance of U-KAN, on this specific dataset, is slightly lower than that of U-Net (which is a strong baseline in registration as we proved in our workshop paper (LKU-Net)[https://github.com/xi-jia/LKU-Net]). However, we found that U-KAN actually takes longer to converge, therefore, we believe the performance of U-KAN might be easily improved by adopting more iterations.

* The current model is limited to 40 (ES-ED) cardiac 2D image pairs and trained on CPU only, due to lacking GPU resources. We believe large-scale experiments are needed for a more comprehensive evaluation of U-Net and U-KAN.

## Results

We note that for a fair comparison, our U-KAN did not use the base_conv option, as discussed in https://github.com/ZiyaoLi/fast-kan/issues/8

* For a fair comparison, we trained 3 models with 3 different manual seeds. Currently, the average registration performance of U-KAN is slightly lower than that of U-Net.
* We used only the mid slice from ES and ED to ease the training burden under CPU. 
* The results for U-KAN might be further improved by replacing RBF with different base functions such as B-Spline.

|                 | Seed 0        | Seed 1        | Seed 2        |
|-----------------|---------------|---------------|---------------|
| U-Net-4         | 0.813 ± 0.078 | 0.836 ± 0.056 | 0.805 ± 0.079 |
| U-KAN-4-RBF     | 0.804 ± 0.072 | 0.823 ± 0.057 | 0.806 ± 0.064 |
| U-Net-8         | 0.851 ± 0.052 | 0.855 ± 0.056 | 0.855 ± 0.047 |
| U-KAN-8-RBF     | 0.822 ± 0.067 | 0.818 ± 0.063 | 0.817 ± 0.064 |
| U-KAN-8-BSpline | 0.833 ± 0.066 | -             | -             |


## Reproducibility


* Training

```
cd ./U-KAN/

python train.py --global_seed 0 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
python train.py --global_seed 1 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
python train.py --global_seed 2 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
```

* Testing


```
cd ./U-KAN/

python test.py --global_seed 0 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
python test.py --global_seed 1 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
python test.py --global_seed 2 --start_channel 4 --using_l2 1 --smth_labda 0.05 --lr 1e-4 --iteration 40001 --checkpoint 400 --net_activation BSpline
```
