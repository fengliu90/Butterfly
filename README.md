# Butterfly for Wildly Unsupervised Domain Adaptation
This is the official site for the paper "Butterfly: One-step Approach towards Wildly Unsupervised Domain Adaptation" (http://128.84.4.34/pdf/1905.07720). This is a joint work with 

Prof. Jie Lu (UTS)

Dr. Bo Han (HKBU/RIKEN)

Dr. Gang Niu (RIKEN)

A/Prof. Guangquan Zhang (UTS)

Prof. Masashi Sugiyama (UTokyo/RIKEN).

This paper has been accepted by NeurIPS'19 LTS Workshop and under the second review of IEEE-TPAMI.

# Software version
TensorFlow version is 1.14.0. Python version is 3.7.3. CUDA version is 11.2.

These python files require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.7.3), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda, tensorflow and download MNIST and SYND data below, you can run codes successfully. Good luck!

# Data download
You can download MNIST and SYND data used in our paper from https://drive.google.com/file/d/1f4sepb0JXeSSftfAWd6KPa_AVoWA28rj/view?usp=sharing.

You can download Amazon and BCIS data used in our paper from https://drive.google.com/file/d/1hSXn8ctm2vMHu_iIhVS_xB4cq2JV34na/view?usp=sharing.

Ensure that you have the following folder structure after downloading:

'./data/synth_train_32x32.mat'

'./data/synth_test_32x32.mat'

'./data/train_mnist_32x32.npy'

'./data/test_mnist_32x32.npy'

'./data/amazon.mat'

'./data/dense_setup_decaf7/dense_bing_decaf7.mat'

'./data/dense_setup_decaf7/dense_caltech256_decaf7.mat'

'./data/dense_setup_decaf7/dense_imagenet_decaf7.mat'

'./data/dense_setup_decaf7/dense_sun_decaf7.mat'

# Flying Butterfly
There are 8 tasks in Butterfly_total.py file:
1. MNIST to SYND (M2S) under Pair-flip noise 20% (P20)
2. MNIST to SYND (M2S) under Pair-flip noise 45% (P45)
3. MNIST to SYND (M2S) under Symmetry-flip noise 20% (S20)
4. MNIST to SYND (M2S) under Symmetry-flip noise 45% (S45)
5. SYND to MNIST (S2M) under Pair-flip noise 20% (P20)
6. SYND to MNIST (S2M) under Pair-flip noise 45% (P45)
7. SYND to MNIST (S2M) under Symmetry-flip noise 45% (S45)
8. SYND to MNIST (S2M) under Symmetry-flip noise 45% (S45)

You can specify one task and run Butterfly using 

python Butterfly_total.py --Task_type 1 --> get results for the first task.

There are 24 tasks in ButterflyNet_Amazon_data.py file. You can run 

python ButterflyNet_Amazon_data.py --> get results for Amazon dataset.

There are 3 tasks in ButterflyNet_Obj_data.py file. You can run 

python ButterflyNet_Obj_data.py --> get results for BCIS dataset.
