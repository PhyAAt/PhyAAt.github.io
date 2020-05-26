---
layout: base
---

# A Quick Start with SVM [~3min]
In this notebook, we explain to download the dataset and getting started with all the predictive tasks using Support Vector Machine. We will be extracting spectral features, specifically 6 rhythmic features - total power in 6 frequency bands, namely, Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-14 Hz), Beta (14-30 Hz), Low Gamma (30-47 Hz), and High Gamma (47-64 Hz). For preprocessing, we filter EEG first with 0.5 Hz highpass and then remove Artifact with ICA based approach.

<p style="text-align:right; font-weight:bold;">Execute with <br><a class="reference external image-reference" href="https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?filepath=PhyAAt_Notebooks/Example0_QuickStartSVM.ipynb" target="_blank"><img src="https://mybinder.org/badge_logo.svg" width="150px"></a></p>

## Complete code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

#!pip install phyaat  # if not installed yet
import phyaat
print('Version :' ,phyaat.__version__)
import phyaat as ph

# Download dataset of one subject only (subject=1)
# To download data of all the subjects use subject =-1 or for specify for one e.g.subject=10

dirPath = ph.download_data(baseDir='../PhyAAt_Data', subject=1,verbose=0,overwrite=False)

baseDir='../PhyAAt_Data'   # or dirPath return path from above

#returns a dictionary containing file names of all the subjects available in baseDir
SubID = ph.ReadFilesPath(baseDir)

#check files of subject=1
print(SubID[1])

 # Create a Subj holding dataset of subject=1
Subj = ph.Subject(SubID[1])


#filtering with highpass filter of cutoff frequency 0.5Hz
Subj.filter_EEG(band =[0.5],btype='highpass',order=5)


# Extract Rhythmic features for task=4
X_train,y_train,X_test, y_test = Subj.getXy_eeg(task=4)


print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nClass labels :',np.unique(y_train))

# Normalization - SVM works well with normalized features
means = X_train.mean(0)
std   = X_train.std(0)
X_train = (X_train-means)/std
X_test  = (X_test-means)/std


# Training
clf = svm.SVC(kernel='rbf', C=1,gamma='auto')
clf.fit(X_train,y_train)

# Predition
ytp = clf.predict(X_train)
ysp = clf.predict(X_test)

# Evaluation

print('Training Accuracy:',np.mean(y_train==ytp))
print('Testing  Accuracy:',np.mean(y_test==ysp))

```

PhyAAt Processing lib Loaded... <br>
Version : 0.0.2

Total Subjects :  1#######################################] S1

{'sigFile': '../PhyAAt_Data/phyaat_dataset/Signals/S1/S1_Signals.csv', 'txtFile': '../PhyAAt_Data/phyaat_dataset/Signals/S1/S1_Textscore.csv'}

100%|##################################################|100\100|Sg - 0 <br>
Done.. <br>
100%|##################################################|100\100|Sg - 1 <br>
Done.. <br>
100%|##################################################|100\100|Sg - 2 <br>
Done.. <br>
100%|##################################################|43\43|Sg - 0 <br>
Done.. <br>
100%|##################################################|43\43|Sg - 1 <br>
Done.. <br>
100%|##################################################|43\43|Sg - 2 <br>
Done.. <br>
DataShape:  (290, 84) (290,) (120, 84) (120,) <br>

Class labels : [0 1 2] <br>
Training Accuracy: 0.9310344827586207 <br>
Testing  Accuracy: 0.8666666666666667




<p style="text-align:center; font-weight:bold;">Execute with <br><a class="reference external image-reference" href="https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?filepath=PhyAAt_Notebooks/Example0_QuickStartSVM.ipynb" target="_blank"><img src="https://mybinder.org/badge_logo.svg" width="150px"></a></p>
