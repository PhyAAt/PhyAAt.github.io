---
title: deep learning
layout: base
---



# Machine Learning Models

## Support Vector Machine


```python
import numpy as np
import matplotlib.pyplot as plt


import phyaat as ph
from sklearn import svm


# to download dataset of subject 1 in given path 'dirpath'
dirPath = ph.load_dataset(dirpath = '/phyaat/dataset/',subject=1)


Subj1  = ph.read_data(dirPath, S=1)
Subj1.filterEEG(band = [0.5],btype='highpass')
Subj1.RemoveArtifact(method='ICA')
X_train,y_train,X_test,y_test = SUbj1.getXy(feature='rhythmic',task=1,split='Sequential')

means = X_train.mean(1)
std = X_train.std(1)

X_train = (X_train-means)/std
X_test = (X_test-means)/std



clf = svm.SVC(kernel='rbf', C=1)

clf.fit(X_train,y_train)

ytp = clf.predict(X_train)
ysp = clf.predict(X_test)

print('Training Accuracy:',np.mean(y_train==ytp))
print('Testing  Accuracy:',np.mean(y_test==ysp))

```

# Deeplearning Models

## Convolutional Neural Network

## Long-short Term Memory - RNN
