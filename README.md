# Action Recognition in Videos through a Transfer Learning based Technique 
By:  López-Lozada, E.; Sossa, H.; Rubio-Espino, E.; Montiel-Pérez, J. Y.

## Introduction 
##---


In computer vision, human action recognition is a hot topic, popularized by the development of deep learning. Since transfer learning based techniques allow to reuse what other models have already learned and to train models with less computational resources, in this work we propose to use a transfer learning based approach for action recognition in videos. We proposed a methodology for human action recognition using transfer learning techniques on a custom dataset. The proposed methodology consists of four steps: 1) human detection and tracking, 2) video preprocessing, 3) feature extraction, and 4) action recognition. This repository presents the software used for the development of such methodology.

<img title="Proposed method" alt="Alt text" src="/images/general_pipeline_proposed_method_mdpi.png">



<img title="Human tracking" alt="Alt text" src="/images/frame_cropped.png">

<img title="Human tracking" alt="Alt text" src="/images/preprocess.png">

## Requirements
It is recommended that you work in a virtual environment. We worked with virtualenv. You will also need to install pytorch and tensorflow.

#### 1. Creación de ambiente virtual 

    virtualenv mdpi -p python3.12.3 
     

#### 2. Inicialización del ambiente virtual 

    source mdpi/bin/activate
     

#### 3. Instalación de paquetes 

    pip install opencv-python 

    pip install matplotlib seaborn 

    pip install cython 

    pip install cython-bbox 

    pip install motmetrics

 
#### Instalación de dependencias para FAIRMOT 

    git clone https://github.com/lucasjinreal/DCNv2_latest.git 

    cd DCNv2_latest/ 

    python setup.py build develop

## Training
For training, there is needed to process all the video frames such as it is mencioned in the work. Then, in the file tf_fine_tunning_test_opFlow_rgb.py modify "ds_file" the directory where the csv with videos are allocated.

    python tf_fine_tunning_test_opFlow_rgb.py 

## Dataset
Data is available at the following link. Note that the raw data taken from the [NTU RGB+D dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)  must be downloaded from the official site, especially the videos of the fall and drinking classes. Data are divided into the folders of processed RGB and motion data and raw data. 

[Download Data](https://drive.google.com/drive/folders/1CA8kNgbqTSdoYop1se3rd667aw7QQuYn?usp=sharing)

## Citation 
López-Lozada, E.; Sossa, H.; Rubio-Espino, E.; Montiel-Pérez, J. Y. Action Recognition in Videos Through a Transfer Learning Based Technique. Preprints 2024, 2024061670. https://doi.org/10.20944/preprints202406.1670.v1

## References

- Zhang, Y.; Wang, C.; Wang, X.; Zeng, W.; Liu, W. FairMOT: On the Fairness of Detection and Re-identification in Multiple Object Tracking. International Journal of Computer Vision 2021, 129, 3069–3087. https://doi.org/10.1007/s11263-021-01513-4.
- Contributors, M. OpenMMLab Pose Estimation Toolbox and Benchmark. https://github.com/open-mmlab/mmpose, 2020. 
