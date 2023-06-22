---
title: visualization
layout: base
---


  <div class="section" id="experiment">
  <h1>Visualization<a class="headerlink" href="#experiment" title="Permalink to this headline">¶</a></h1>
  <p>There are a few different approaches to visualise the EEG Data. One of the most interesting and cool visualization technique is to view the brain activity as topographical map. </p>
  <!--<li><a class="reference external" href="http://nikeshbajaj.in">Nikesh Bajaj<img alt="Nikesh Bajaj" src="_images0/nikeshbajaj.png" width="100"></a></li> -->
  <div class="section" id="institutions">
  <h2>Topographical Map<a class="headerlink" href="#institutions" title="Permalink to this headline">¶</a></h2>
Topographical map display the activity of brain on a 2D circular picture as to maintained the spatial realtionship of electrodes. The values are shown as heat-map, which could represent firing rate of neurons at the given site, spectral power or total energy of signal, or some other measure.

Following figure shows Brain activity as measured by energy of signal at given sites. The dynamics of the activity is measured by computing energy of signals over 1 sec window and shifting it by 0.5 sec. 

Top left figure shows the activity extracted from raw EEG, and other figures shows corresposing brain activity in different frequency bands.
    
  <a><img alt="Under construction" src="{{ "https://raw.githubusercontent.com/Nikeshbajaj/EEG_Visualization/master/Figures/AllTogether.gif" | relative_url }}" width="100%"></a>
  </div>
  </div>


To generate such plots for static or dynamic measurements, python library - [spkit](https://spkit.github.io/) is used. For details check following link - [**https://spkit.github.io/examples/gen_topo**](https://spkit.github.io/examples/gen_topo)


