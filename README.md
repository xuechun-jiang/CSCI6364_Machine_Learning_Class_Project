# CSCI6364 Machine Learning Class Project  
**A Comparative Study of Bird Species Classification Models**

## Overview

This repository contains the implementation and experimental code for the course project of **CSCI 6364 – Machine Learning**.  
The project investigates a **fine-grained bird species classification task under data-scarce conditions**, and systematically compares three different modeling strategies:

- Training a **custom convolutional neural network (CNN) from scratch**
- **Fine-tuning pre-trained deep CNN models**
- Combining **deep feature extraction with a Support Vector Machine (SVM) classifier**

In total, **11 different models** are implemented and evaluated to analyze classification performance, convergence behavior, and generalization ability.

---

## Dataset

We use a subset of the **CUB-200-2011** bird dataset.  
To focus on a challenging fine-grained classification setting, we manually selected **19 visually similar bird species**, which are difficult to distinguish due to overlapping appearance features such as plumage color, crest shape, and body structure.

### Selected Bird Species (19 Classes)

- Black_footed_Albatross  
- Bobolink  
- Bohemian_Waxwing  
- Brewer_Blackbird  
- Gray_Catbird  
- Gray_crowned_Rosy_Finch  
- Groove_billed_Ani  
- Indigo_Bunting  
- Laysan_Albatross  
- Worm_eating_Warbler  
- Least_Flycatcher  
- Louisiana_Waterthrush  
- Red_bellied_Woodpecker  
- Red_eyed_Vireo  
- Red_headed_Woodpecker  
- Red_winged_Blackbird  
- Rusty_Blackbird  
- Tennessee_Warbler  
- Wilson_Warbler  

### Data Split

For each species, **59 images** are used:

- **Training:** 40 images  
- **Validation:** 7 images  
- **Test:** 7 images  

This results in:

- Training set: **720 images**  
- Validation set: **133 images**  
- Test set: **133 images**  

The limited size of the training data is intentional and is used to evaluate model performance under **low-data learning conditions**.

---

## Project Structure

```text
CSCI6364_Machine_Learning_Class_Project/
│
├── Data/                       # Dataset (train / valid / test)
├── image/                      # Figures (accuracy curves, confusion matrices)
├── tools/                      # Data loaders, evaluation, utilities
│
├── customized_NN.py            # Custom CNN trained from scratch
├── fine_tuned_resnet.py        # Fine-tuned ResNet models
├── fine_tuned_densenet.py      # Fine-tuned DenseNet-121
├── fine_tuned_efficientnet.py  # Fine-tuned EfficientNet-B0
├── SVM.py                      # Deep feature extraction + SVM
│
└── README.md
