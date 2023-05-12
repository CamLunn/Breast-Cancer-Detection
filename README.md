# Breast-Cancer-Detection
Machine Learning - Group 9 Final Project

Contents:
- **Code**: folder containing a Classification subfolder and a MaskRCNN subfolder. 
  -  The Classification folder the jupyter notebook file used for the classificatoin using CNN, AlexNet, PCA-SVM hybrid, and SVM.
  -  The MaskRCNN folder contains all code required to reproduce the segmentation results.
- **Dataset/csv**: folder containing the annotations.csv file used for reading the data in. *Note: see Reproduce Results section at bottom to retrieve data.*
- **Resources**: folder containing the writeup and powerpoint presentation used for 



## Introduction

As one of the leaders in cancer mortality, breast cancer ranks fifth worldwide in causes for death in females [18]. In the U.S. alone, approximately 287,850 new cases of invasive breast cancer and 51,400 cases of DCIS (ductal carcinoma in situ) are diagnosed, and 43,250 died from breast cancer [1]. Moreover, it does limit itself to the female gender; males also have this sort of cancer, though the cases reported every year are nowhere near that of women cases and deaths [2]. This project aims to support the early screening and diagnosis of breast cancer through mammograms. The early detection of breast cancer contributes significantly to reducing the death rate. For this purpose, many screening methods are used, like ultrasound, screen-film mammography, magnetic resource imaging, and digital mammography [3]. Mammography is an X-ray technique that was developed specifically for breast lesion examination. Diagnosis, evaluation, and determination of the results are based on the different absorption of X-rays between different types of breast tissue [4]. A few of the injuries (small lesions) in mammograms may go undetected or be analyzed erroneously due to the quality of mammograms, the inability (experience) of the radiologists, or the limitation within the human visual system.

To bypass the issue, Radiologists use computer-aided detection/diagnosis systems (CAD) for breast cancer detection; boosting the accuracy of the CAD system can improve detection accuracy, and this will end in a better survival rate and treatment choices. CAD comprises a fundamental two-phase segmentation and classification; segmentation is an essential step in a computer-aided detection/ diagnostic (CAD) system; handcraft segmentation methods are not giving the precision recommended, are time-consuming, and are very tedious.

The tremendous progress in artificial intelligence, especially computer vision, has contributed significantly to vision systems development in various fields, such as autonomous driving [5], face recognition [6], handwriting recognition, and healthcare systems [7]. Motivated by this progress, this project utilizes deep learning to address Breast Cancer Detection. Specifically, Convolutional Neural Networks (CNN) and SVM are implemented for classification of masses, and MaskRcnn for segmentation on mammograms. The overall framework comprises three parts: 1) Data Pre-processing, 2) Classification model (CNN + SVM), and 3) Detection-segmentation based on the Mask R-CNN [8]. The overall architecture provides a flexible and efficient classification, object detection, and segmentation framework. For segmentation, after lesion extraction, it is further segmented, and the result is compared to the ground truth. The system is evaluated on CBIS-DDSM [9], a publicly available benchmark.



## Model Architectures used for classification:

1. Custom AlexNet
<img width="346" alt="Custom_AlexNet" src="https://github.com/CamLunn/Breast-Cancer-Detection/assets/64609764/a5ef334c-900b-4e3b-b51c-919c5fe843db">

2. Fine-tuned AlexNet
<img width="544" alt="Fine-tuned_AlexNet" src="https://github.com/CamLunn/Breast-Cancer-Detection/assets/64609764/d4d7c264-74ce-4b6b-8944-5d945192558f">

3. 7-layer AlexNet using PCA-SVM, and SVM classifiers.
<img width="848" alt="complete_full" src="https://github.com/CamLunn/Breast-Cancer-Detection/assets/64609764/965c67a7-e95c-49e3-8f76-8fca4d271789">

4. 6-layer AlexNet using PCA-SVM, and SVM classifiers.
<img width="853" alt="complete_truncated" src="https://github.com/CamLunn/Breast-Cancer-Detection/assets/64609764/496f4ca6-bb50-4f51-948a-b01af115e690">


## Reproduce results
To reproduce our results, you'll need to first download the data from https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset and unzip it in the Dataset directory


