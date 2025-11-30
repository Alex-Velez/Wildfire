Image Dataset 1: `jafar_2023`
> Ibn Jafar, Anam; Islam, Al Mohimanul ; Binta Masud, Fatiha; Ullah, Jeath Rahmat; Ahmed, Md. Rayhan (2023), “FlameVision : A new dataset for wildfire classification and detection using aerial imagery ”, Mendeley Data, V4. https://doi.org/10.17632/fgvscdjsmt.4

Image Dataset 2: `madafri_2023`
> El-Madafri I, Peña M, Olmedo-Torre N. The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach. Forests. 2023; 14(9):1697. https://doi.org/10.3390/f14091697

Image Dataset 3: `aaba_2022`
> Aaba, A. (2022). Wildfire Prediction Dataset (Satellite Images) [Data set]. Kaggle. https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data

Image Dataset 4: `xu_2024`
> Xu, Y., Berg, A., & Haglund, L. (2024, March 26). Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data. https://doi.org/10.5281/zenodo.10881058 

# Data Folder Structure
```
##################################################
#                                                #
#  When you download all datasets:               #
#    1. Renamed parent folders `lastname_year`.  #
#    2. Keep the file structure unmodified.      #
#                                                #
##################################################

Wildfire
├───wildfire.ipynb
└───data
    ├───aaba_2022
    │   ├───test
    │   │   ├───nowildfire
    │   │   └───wildfire
    │   ├───train
    │   │   ├───nowildfire
    │   │   └───wildfire
    │   └───valid
    │       ├───nowildfire
    │       └───wildfire
    ├───jafar_2023
    │   ├───Classification
    │   │   ├───test
    │   │   │   ├───fire
    │   │   │   └───nofire
    │   │   ├───train
    │   │   │   ├───fire
    │   │   │   └───nofire
    │   │   └───valid
    │   │       ├───fire
    │   │       └───nofire
    │   └───Detection
    │       ├───test
    │       │   ├───annotations
    │       │   └───images
    │       ├───train
    │       │   ├───annotations
    │       │   └───images
    │       └───valid
    │           ├───annotations
    │           └───images
    ├───madafri_2023
    │   ├───test
    │   │   ├───fire
    │   │   └───nofire
    │   ├───train
    │   │   ├───fire
    │   │   └───nofire
    │   └───val
    │       ├───fire
    │       └───nofire
    └───xu_2024
        ├───scene1
        ├───scene2
        ├───scene3
        └───scene4
'''