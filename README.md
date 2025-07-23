# perceptual-quality-classification

Single-file implementation of the experiments from  
“Examining the Role of Perceptual Quality in Underwater Image Classification”  
paper link → https://ieeexplore.ieee.org/document/10970163  

The GitHub repo contains only `Train_Test_.py`.  
All data (images + CSV label files) is delivered through Google Drive.

---

## 1. Download the prepared dataset

Google-Drive folder (download everything) → **https://drive.google.com/drive/folders/1ReinQ7FGpP1uSyQ5T3u4R3Kk1ZEFpqqp?usp=sharing**

After unzipping you must see the following items:

* GOOD_QUALITY_TRAINING/          → high-clarity training images  
* POOR_QUALITY_TRAINING/          → low-clarity training images  
* COMBINED_QUALITY_TRAINING/      → both sets merged  
* Test_Images/                    → good-quality test images  
* P_Test_Images/                  → poor-quality  test images  
* Test_data_classes.csv           → labels for Test_Images  
* P_Test_data_classes.csv         → labels for P_Test_Images  
* Train_Test_.py                  → same script that lives in this repo  

---

## 2. What the script does by default

1. Uses **ResNet50** (ImageNet weights) as the backbone.  
2. Trains on images inside `GOOD_QUALITY_TRAINING/`.  
3. Evaluates each epoch on two separate test sets:  
   * good quality → `Test_Images/` + `Test_data_classes.csv`  
   * poor quality → `P_Test_Images/` + `P_Test_data_classes.csv`  
4. Repeats the entire training process three times (`n_models = 3`)
5. Keeps the best epoch weights for each training process (highest accuracy on the good-quality test set).  
6. Saves per-epoch metrics to CSV


## 3. Lines you may want to edit in `Train_Test_.py`

Search for these lines and follow the comments next to them.

* **Base model selection** (Uncomment exactly one line to pick another backbone)

from tensorflow.keras.applications import ResNet50        as BaseModel
from tensorflow.keras.applications import MobileNetV2      as BaseModel
from tensorflow.keras.applications import DenseNet121      as BaseModel
from tensorflow.keras.applications import EfficientNetB0   as BaseModel
from tensorflow.keras.applications import EfficientNetV2B3  as BaseModel





* **Training folder**  

Change the string to  
`'GOOD_QUALITY_TRAINING'`, `'POOR_QUALITY_TRAINING'`, or `'COMBINED_QUALITY_TRAINING'`  
depending on which subset you want.


If you use this code or the accompanying data, please cite our work:

> D. S. Almeida, E. Pedrosa and N. Lau, "Examining the Role of Perceptual Quality in Underwater Image Classification," 2025 IEEE International Conference on Autonomous Robot Systems and Competitions (ICARSC), Funchal, Portugal, 2025, pp. 1-8, doi: 10.1109/ICARSC65809.2025.10970163.


Our dataset originates from the EUVP Dataset.
Link to the original EUVP Dataset  →  https://irvlab.cs.umn.edu/resources/euvp-dataset
