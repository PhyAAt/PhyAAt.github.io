---
title: introduction
layout: base
---

# Introduction:  Getting started
*library will be uodated soon...*


In this section, we explain how to get started with the dataset and modeling. For ease all the necessary element and codes are put into one python library called - ***phyaat***


<h2 class="no-bg">1. Install Library </h2>

First install the python library

```console
pip install phyaat
```

<h2 class="no-bg">2. Download dataset </h2>
Once Phyaat library is installed, the dataset can be downloaded using it. You could download all the dataset together or data of one particulat subject for testing and running.

```python
import phyaat as ph

# to download dataset of subject 1 in given path 'dirpath'
dirPath = ph.load_dataset(dirpath = '/phyaat/dataset/',subject=1)

# to download dataset of all the subjects
dirPath = ph.load_dataset(dirpath = '/phyaat/dataset/',subject=-1)

```

<h2 class="no-bg">3. Preprocessing </h2>

<h4 class="no-bg">3.1. Filtering </h4>

```python
import phyaat as ph


```


<h4 class="no-bg">3.2 Applyting Artifact Removal on EEG </h4>

```python
import phyaat as ph


```

<h2 class="no-bg">4. Extract X,y for task</h2>

<h4 class="no-bg">4.1 Extracting Segments - listening, writing, resting</h4>

```python
import phyaat as ph


```


<h4 class="no-bg">4.2 Extracting X (raw signals), y for given tasks </h4>
```python
import phyaat as ph


```


<h4 class="no-bg">4.3 Extracting X (spectral features), y for given tasks </h4>

```python
import phyaat as ph


```


<h2 class="no-bg">5. Applying pretive models - SVM</h2>

```python
import phyaat as ph


```
