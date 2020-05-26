---
title: experimenet
layout: base
---

<div class="section" id="experiment"></div>
  <!--<h1>Under construction...<a class="headerlink" href="#experiment" title="Permalink to this headline">¶</a></h1> -->

# Experiment paradigm
*Due to relevent article under double blind review, contents are not fully updated here, will be uodated soon..*
<div style="text-align: justify">
The experiment for recording the physiological signals for auditory attention is based on widely used experimental setting in cognitive psychology - diachotic listening task. Here we will provide a very high level explaination of the experiment, for details please refere to the article here <a href="https://arxiv.org/pdf/2005.11577.pdf" target="_blank"> <i class="fa fa-file-pdf-o" style="font-size:24px;color:red"></i></a>. Three physiological signals (<strong>EEG, GSR, PPG</strong>) were recorded from 25 healthy subjects, while conduting the experiment. The collected dataset is carefully labeled and four predictive tasks are formulated. Estimating <strong>Auditory Attention level</strong> from physiological signals is one of them.
</div>

## Experiment Design
<div style="text-align: justify"></div>
The experiment is based on listening task, unlike diachotic listening task, each subject was presented only one auditory stimulus under different auditory conditions for 1 trial. As shown in Figure 1. The auditory conditions include different level of background noise (**N**), Sementicity (**S**) and length (**L**) of audio stimulus. Each subject was presented with 144 stimuli, one per trial, with no repeatition of audio message. The order of stimuli with different auditory conditions were randomized. The physiological signals (**R**) were recorded at sampling rate of 128 Hz.

<img class="center" alt="Experimental Model" src="{{ "/assets/images/ExperimentModel.png" | relative_url }}" width="80%">
<figcaption>Fig. 1: Experiment Design</figcaption>

For computing the **"Auditory Attention Score (A)"**, following the literature, the number of correctly identetified words $$N_C$$) in the trasncriped audio message were counted and Attention Score $$ A = \frac{N_C}{N_T}\times 100$$, computed where $$N_T$$ is total words in the original audio message.



## Experimental procedure
<div style="text-align: justify"></div>
As shown in Figure 2 below (in right), each trial consists of three tasks, listerning, writing, and resting. Using a computer interfact, A subject can actively choose to play an audio stimulus while listerning task. Once audio is finished, subject needs to transcribe the message (writing task). Audio can not be reproduced (replayed). Once transcription is done, writing task can be finished by submit button. The duration between writing task and next listening task is labeled as resting task. On the average total time taken by one subject was $$40\pm10$$. The pitcute of a participent performing the experiment is shown below (on the left side).

<img class=""  alt="Experimental Model" src="{{ "/assets/images/Expic.jpeg" | relative_url }}" width="50%"><img class="" alt="Experimental Model" src="{{ "/assets/images/ExTimeline.png" | relative_url }}" width="50%">
<figcaption>Fig. 2: left: participant performing experiment, right: timeline of one trial.</figcaption>


## Predictive tasks
Following four preditive tasks are formulated. The details of formulation and respective justification is explained in the paper.

$$
\begin{eqnarray}
\text{T1: Attention Score prediction} &:& A^{\prime} = f_A(F_r)\\
\text{T2: Noise Level prediction} &:& N^{\prime} = f_N(F_r)\\
\text{T3: Semanticity prediction} &:& S^{\prime} = f_S(F_r)\\
\text{T4: LWR Classification} &:& \mathcal{T}^{\prime} = f_{\mathcal{T}}(F_r)
\end{eqnarray}
$$

where $$F_r$$ is feature vector extracted from Physiological Responses $$R$$: $$R \rightarrow F_r$$



## Participents - demographics
*will be uodated soon..*

## Collected Dataset
For details about collected dataset - please see [**here**](/dataset)
