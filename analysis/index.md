---
title: analysis
layout: base
---

# Analysis

## Text response analysis
Statistical analysis of the text response was done in great detail. Article for findings is published and in referered as below.

* **Analysis of Factors Affecting the Auditory Attention of Non-native Speakers in e-Learning Environments**, The Electronic Journal of e-Learning, 19(3), pp. 159-169.
<a href="https://academic-publishing.org/index.php/ejel/article/view/2296" target="_blank"> <i class="fa fa-file-pdf-o" style="font-size:24px;color:red"></i></a> [PDF](https://academic-publishing.org/index.php/ejel/article/view/2296)
  

**Interaction between Noise and Semanticity over Attention Score.**  

One of the most interesting findings we had from this study was a dimishing gap of attenstion level of between semantic vs non-semnatic stimuli (as shown below in left). 

<center>
<img alt="" src="{{ "analysis/figures/NvsSem_3.png" | relative_url }}" width="40%">
<img alt="" src="{{ "analysis/figures/NvsL_3.png" | relative_url }}" width="40%">
<figcaption>Fig. 1: Interaction plot: Left - Attention level vs Noise with semantic and non-semantic stimuli. Right - Attention level vs Noise with different length of stimuli</figcaption>
</center>


**Clustering groups of experimentation conditions based on attention scores.**

This figure shows the hierarchical clustering analysis of all the experimental conditions with attention score.
<center>
<img alt="" src="{{ "analysis/figures/Pmatrix005_ann.png" | relative_url }}" width="60%">
<figcaption>Fig. 2: Clustering Analysis</figcaption>
</center>


* For more detail, please check the paper <a href="https://academic-publishing.org/index.php/ejel/article/view/2296" target="_blank"> <i class="fa fa-file-pdf-o" style="font-size:24px;color:red"></i></a> 
* **To download the tabular data for statistical analysis check here** 
* [**Download tabular data**](https://phyaat.github.io/dataset#tabular-data)


## Signal analysis - for feature engineering

Some of the analysis done on the signals are presented in the paper: <strong>Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks</strong>. <i>Biomedical Signal Processing and Control</i>, 55, 101624.<a href="https://doi.org/10.1016/j.bspc.2019.101624" target="_blank"> <i class="fa fa-file-pdf-o" style="font-size:24px;color:red"></i> </a><strong><a class="reference external" href="https://doi.org/10.1016/j.bspc.2019.101624" target="_blank">[PDF]</a></strong>


### Spectral Analysis

One of the most fundamental analysis done on physiological data, specifically EEG is spectral analysis. In these figures, the affect of artifact removal algorithm is shown.

<center>
<img alt="Spectral Analysis" src="{{ "/assets/images/PSD.png" | relative_url }}" width="40%">
<img alt="Spectral Analysis" src="{{ "/assets/images/Spectral_LWR_Algo.png" | relative_url }}" width="40%">
<figcaption>Fig. 3:Spectral Analysis of LWR</figcaption>
</center>

### Statistical analysis

This figure shows, how distribution of EEG signal changes with ATAR algorithm.
<center>
<img class="center" alt="Distribuation Analysis" src="{{ "/assets/images/PDF2.png" | relative_url }}" width="40%">
<figcaption>Fig. 4: Ditribuation of signals</figcaption>
</center>

### Event Related Pontential analysis

All the event related analysis are plotted as a structure shown below, which places the ERP analysis of a electrode in its repective location with respect to topographical arangement.

<center>
<img class="center" alt="ERP Analysis" src="{{ "analysis/figures/S5_ERP_lwr_circ.png" | relative_url }}" width="30%">
<figcaption>Fig. 5: </figcaption>
</center>




#### ERP Analysis of Semanticity
<center>
<img  alt="ERP Analysis" src="{{ "analysis/figures/S5_ERP_Sm.png" | relative_url }}" width="70%">
<figcaption>Fig. 6: ERP Analysis of Semanticity</figcaption>
</center>

#### ERP Analysis of Noise Level
<center>
<img class="center" alt="ERP Analysis" src="{{ "analysis/figures/S5_ERP_noise.png" | relative_url }}" width="70%">
<figcaption>Fig. 7: ERP Analysis of Noise Level</figcaption>
</center>

#### ERP Analysis of LWR Task
<center>
<img  alt="ERP Analysis" src="{{ "analysis/figures/S5_ERP_lwr.png" | relative_url }}" width="70%">
<figcaption>Fig. 8: ERP Analysis of LWR task</figcaption>
</center>