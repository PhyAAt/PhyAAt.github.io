---
title: introduction
layout: base
---

# Introduction:  Getting started

In this section, we explain how to get started with the dataset and modeling. For ease all the necessary element and codes are put into one python library called - ***phyaat***. Here we explain the functionalities that Phyaat library has with possible tuning the process of preprocessing and feature extractions. To start with a quick exmaple to preditive modeling check the [**Predictive Modeling**  ](/modeling) tab.

<font size="4"> For quick start with predictive modeling, check <a href="/modeling/index.html" target="_blank"> <span style="font-weight:bold"> EXAMPLE CODES</span></a></font>


<h2 class="no-bg">1. Install Library </h2>

First install the python library

```console
pip install phyaat
```

<h2 class="no-bg">2. Download dataset </h2>
Once Phyaat library is installed, the dataset can be downloaded using it. You could download all the dataset together or data of one particulat subject for testing and running.

```python
import phyaat
print('Version :' ,phyaat.__version__)
import phyaat as ph
```


### To download dataset of subject 1'
To download data set of only one subject with subject id=1 (subject=1), use following code. Here baseDir = '../PhyAAt_Data' is path where data will be downloaded and stored. Make sure you have permission to write in given path.

```python
dirPath = ph.download_data(baseDir='../PhyAAt_Data', subject=1,verbose=0,overwrite=False)
#returns a dictionary containing file names of all the subjects available in baseDir
SubID = ph.ReadFilesPath(dirPath)

# list of all the subjects in the dataset directory
print(SubID.keys())
```

### To download dataset of all the subjects
```python
dirPath = ph.download_data(baseDir='../PhyAAt_Data', subject=-1,verbose=0,overwrite=False)

#Check the humber of subjects are in directory - read the file path of all the subjects available

baseDir='../PhyAAt_Data'   # or dirPath return path from above

#returns a dictionary containing file names of all the subjects available in baseDir
SubID = ph.ReadFilesPath(baseDir)
# list of all the subjects in the dataset directory
print(SubID.keys())
```

<h2 class="no-bg">3. Preprocessing </h2>

```python

#Creat an object holding data of a subjects

Subj = ph.Subject(SubID[1])
```


<h3 class="no-bg">3.1. Filtering </h4>

### Highpass filter with cut-off frrequency of 0.5Hz

This is very standard to remove any dc component and drift in the signal

```python
#filtering with highpass filter of cutoff frequency 0.5Hz
Subj.filter_EEG(band =[0.5],btype='highpass',order=5)
```

### Filtering with custum range of feequecy should be between 0-64Hz

#### Lowpass filter
```python
#filtering with lowpass filter Delta
Subj.filter_EEG(band =[30],btype='lowpass',order=5)
```
#### Bandpass filter

```python
#filtering with bandpass filter Thata
Subj.filter_EEG(band =[4,8],btype='bandpass',order=5)
```


<h4 class="no-bg">3.2 Applyting Artifact Removal on EEG ICA based approach</h4>

```python
# with window size =1280 (10 sec)
Subj.correct(method='ICA',verbose=1,winsize=128*10)

#method ='WT' or 'ATAR' not implemented yet


#Chnage parameters of ICA based artifact Removal
KurThr = 2
Corr   = 0.8
ICAMed = 'extended-infomax' #picard, fastICA

Subj.correct(method='ICA',winsize=128,hopesize=None,Corr=Corr,KurThr=KurThr,
             ICAMed=ICAMed,verbose=0, window=['hamming',True],
             winMeth='custom')

#check all the parameters here
help(ph.Subject.correct)
```

<h2 class="no-bg">4. Extract X,y for a task Rhythmic Features</h2>

<h4 class="no-bg">4.1 Extracting Features Segment-wise</h4>

```python
# Task 4:  LWR classification
X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=4)

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nClass labels :',np.unique(y_train))

# Task 1: Attention Score Prediction

X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=1)

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nlabels :',np.unique(y_train))


# Task 2: Noise Level Predicition
X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=2)

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nlabels :',np.unique(y_train))


# Task 3: Semanticity Classification
X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=3)

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nClass labels :',np.unique(y_train))

#If features are extracted for task 1, 2 or 3 (listening segments)
# next time while extracting won't compute features again, unless redo=True

```

<h4 class="no-bg">4.1 Extracting Features Window-wise</h4>

```python
 winsize=128 # 1 sec window
 hopesize=32 # 0.25 shift for next window, if None, overlape is half of windowsize

X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=1, features='rhythmic',
                            winsize=winsize, hopesize=hopesize)

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nClass labels :',np.unique(y_train))

```

<h4 class="no-bg">4.2 Random split for train-test</h4>

```python

X_train,y_train, X_test,y_test = Subj.getXy_eeg(task=1, features='rhythmic',
                           winsize=winsize, hopesize=hopesize,split='random')

print('DataShape: ',X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('\nClass labels :',np.unique(y_train))

```


<h4 class="no-bg">4.3 Hyperparameters for feature extraction method </h4>
```python

X_train,y_train, X_test,y_test = Subj.getXy_eeg(ttask=1, features='rhythmic', eSample=[0, 0],
               verbose=1, redo=False, split='serial', splitAt=100, normalize=False,
               log10p1=True, flat=True, filter_order=5, method='welch', window='hann',
               scaling='density', detrend='constant', period_average='mean',
               winsize=-1, hopesize=None)

#Check help
print(ph.Subject.getXy_eeg)
```


<h2 class="no-bg">5.Extracting LWR segments for extranal processing</h2>

```python
L,W,R, Scores, Cols = Subj.getLWR()

```
