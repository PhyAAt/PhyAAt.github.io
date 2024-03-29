---
title: algorithms
layout: base
---

<div class="section" id="Algorithm">
<h1>Algorithms<a class="headerlink" href="#experiment" title="Permalink to this headline"></a></h1>
<h2 ><a href="artifact_algo.html">Automatic and Tuanable Artifact Removal (ATAR) algorithm</a></h2>
<strong>Wavelet based approach</strong>, Article - <strong>Artifact Removal</strong> can be found <a href="https://doi.org/10.1016/j.bspc.2019.101624" target="_blank"><strong>here</strong></a><a href="https://doi.org/10.1016/j.bspc.2019.101624" target="_blank"> <i class="fa fa-file-pdf-o" style="font-size:24px;color:red"></i></a>.
 
A Tutorial of explaining how to remove artifact from EEG can be found: <a href="https://link.medium.com/90mYhbr8Osb" target="_blank"><strong>here</strong></a>
 
<br>

<img class="center" src="{{ "/assets/images/Algorithm_BD1.png" | relative_url }}" width="90%">

<figure>
<!-- <img style="float: left;"  src="{{ "/assets/images/SignalsSeg3_WPD50_a.png" | relative_url }}" width="49%">
<img style="float: right;"  src="{{ "/assets/images/SignalsSeg3_WPD50_b.png" | relative_url }}" width="49%"> -->
 
<img style="float: right;"  src="{{ "/assets/images/ATAR_ICA_1.PNG" | relative_url }}" width="85%">
</figure>

<center>
<figure>
<img style="float: center;"  src="{{ "/assets/images/Beta.gif" | relative_url }}" width="80%">
</figure>
</center>  

 
 
 <br>

<strong>Python implementation </strong> of ATAR Algorithm is now available on <strong>Spkit</strong> library, the examples of which can be found here
 
<ul class="simple">
  <li>* <a href="https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ATAR_Algorithm_EEG_Artifact_Removal.ipynb" target="_blank"><strong>Jupyter Notebook </strong></a></li>
  <li>* <a href="https://spkit.github.io/guide/notebooks/ATAR_Algorithm_EEG_Artifact_Removal.html" target="_blank"><strong>HTML Friendly</strong></a></li>
  <li>* <a href="https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?filepath=spkit/SP/ATAR_Algorithm_EEG_Artifact_Removal.ipynb" target="_blank"><strong>Binder</strong></a></li>
</ul>
Soon will be integrated to phyaat library.


<!--<a><img alt="Under construction" src="../_images0/IPR.gif" width="300"></a>-->
<!--<h3 style="background-color: #EBF5FB">ICA based Artifact removal approach<a class="headerlink" href="#institutions" title="Permalink to this headline"></a></h3> -->

<h2> ICA Based algorithms</h2>
<strong>Python implementation </strong> of ICA based algorithm is now available on <strong>Spkit</strong> library, the examples of which can be found here
<ul class="simple">
<li> * <a href="https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_based_Artifact_Removal.ipynb" target="_blank"> <strong>Jupyter Notebook </strong></a></li>
<li> * <a href="https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?filepath=spkit/SP/ICA_based_Artifact_Removal.ipynb" target="_blank"> <strong>Binder </strong></a></li>
</ul>

<h2>Wavelet based other approachs</h2>
<strong>Wavelet Filtering</strong> can be used to remove the artifact and clean the signal. <strong>Python implementation </strong> of wavelet filtering is available on <strong>Spkit</strong> library, the examples of which can be found here 
<ul class="simple">
<li>* <a href="https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Wavelet_Filtering_1_demo.ipynb" target="_blank"><strong>Jupyter Notebook </strong></a> </li>
<li>* <a href="https://spkit.github.io/guide/notebooks/Wavelet_Filtering_1_demo.html" target="_blank"><strong>HTML Friendly</strong></a> </li>
<li>* <a href="https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?filepath=spkit/SP/Wavelet_Filtering_1_demo.ipynb" target="_blank"><strong>Binder</strong></a> </li>
</ul>
 
 
 
<!--<li><a class="reference external" href="http://nikeshbajaj.in">Nikesh Bajaj<img alt="Nikesh Bajaj" src="_images0/nikeshbajaj.png" width="100"></a></li> -->
