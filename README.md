

# Neural-Network Prediction of Defect Formation in Quenched $\phi^4$ models.

<p align="center">
  <img src="figures/2d/kz_quench.gif" width="1000">
  <br>
  <em>2D quench at different quench times, shown at the same rescaled times t/t̂. Larger τ<sub>Q</sub> gives larger domains.</em>
</p>

## Overview

WHen a system is driven through a symmetry-breaking phase transition at a finite rate, the symmetry in the system is not broken uniformly throughout the system but instead forms domains of broken-symmetry states characterized by defects in the system. The Kibble-Zurek (KZ) mechanism predicts the typical size of these domains, but not any other higher order spatial information such as variance, spatial structure, or defect-defect correlations. It was recently shown in a paper by Suzuki, Li, and Zurek [1] in 1D that a recurrent Neural Network can be trained to predict the defect locations from data deep within the impulse regime, long before the configuration forms. This repository studies this question in more detail and investigates the physics that model learns.


We first replicate the 1D result and show that the predictive information lives in modes at or greater than the Kibble-Zurek length scale. We then extend the experiment to 2D, where domains do not just form, but also coarsen. We demonstrate that given snapshots of the field configuration, a NN designed with the UNET architecture is able to learn the dominant phyiscs of coarsening at early and late timescales, and at intermediate times in evolution is able to learn physics beyond coarsening, decreasing losses by over 20x the coarsening model. The original study in [1] trained on sequential snapshot data, we study the effect of training on instead the single snapshot field configuration $(\phi,\pi)$ and study the effect of including momentum into training as a function of damping parameter, finding that in overly damped systems knowledge of momentum holds little predictive value.

<p align="center">
  <img src="figures/2d/summary.png" width="900">
  <br>
  <em>(a) From an early, noisy field (t = 3 t̂), the U-Net predicts the domain pattern at the end of the run.
  (b) Prediction error vs. input time: the U-Net beats both baselines, most around domain formation (dotted line).
  (c) After formation, the optimal blur width follows the curvature-flow prediction σ² = 2Δt for all quench rates (shaded: before formation).</em>
</p>

# Background

<p align="center">
  <img src="figures/repo_fig/KZM.png" width="700">
  <br>
  <em>The transition of some general disordered model to some crystalline phase. The blue line denotes how long one needs to wait for thermalization. The red lines indicate the cooling rate. The x value at which these two curves intersect is known as the "freezout time" $\hat t$. The classical understanding of KZ mechanism is that dynamics is "frozen out" in the time $t\in [-\hat t , \hat t]$ </em>
</p>

<p align="center">
  <img src="figures/repo_fig/SSB.png" width="700">
  <br>
  <em>The minima of the double-well potential spontaneously takes on a non-zero value as the paramater $\epsilon(t)$ is varied.</em>
</p>
Phase transitions in physical systems are associated with a spontaneous symmetry breaking. For instance, consider the figure above. On the left, we see that the minimum of the curve in red is in the center. Now imagine we deform the curve as shown in the diagram. We see that suddenly, there are now two minima in the function where previously there was only one. 

The red curve represents our systems potential energy. Our system is described by a "state", represented by the blue ball. Imagine we place the ball into the curve and let it go. We see in the first picture on the left that there is only one place we can put it so that the ball does not roll, and that is at the minimum of the potential energy curve. A system sitting at the minimum of a potential enjoys stability. This minimum energy state is called the ground state of the system. 

When there are two degenerate minima, the ball has to pick one to fall in. Both divots lower the balls potential energy by the same amount, so there is priority for which one should be chosen over the other, however a *choice must be made*. While a ball put the left well and a ball put in the right well are going to experience the exact same physics, their configurations are distinguishable (by their $x$ component, for instance). 

If we place the ball at the center of the minimum as in the left case in the figure above, the moment when the potential energy develops two seperate minima the ball must make a choice as to which well it will occupy. This is why we call it spontaneous symmetry breaking, because the symmetry is broken spontaneously! 

The critical point of a system is the point at which symmetry is just about to be broken (e.g. the critical Temperature, critical magnetic field, etc). At this point, the system becomes ultra-sensitive to thermal fluctuations (The situation is slightly different in quantum systems, which we do not consider here). The change in the value of the field at one point has immense influence on the strength of the field at a far away point. To describe this long range sensitivity, we say that the system's *correlation length* "diverges" at the critical point, in other words, we formally say it is infinite. The correlation length of a second order phase transition obeys a scaling law, $\xi \propto \Delta^\nu$, where $\Delta$ is the distance from the critical point and $\nu$ a number greater than 0.

Correlation length can be thought of as the size of an area that has settled into the same value of field strength. However, to equilibriate an area of size $\xi$, one needs to wait a period of time proportional to $\xi^z$. $z$ is the *dynamical exponent* of the system and is always greater than 0. This means that if our correlation length is infinite, then we need to wait an infinite amount of time for our system to equilibriate! 

This implies that any finite speed phase transition (in other words, you drive the system across the critical point in some time that doesn't take an inifnite amount of time) is inherently a non-equilibrium process. Physically, heat energy is injected into the system by the finite speed quench. This heat is observable in the post-quench field configuration as topological defects that raise the energy of the system above its ground state. 

The density of these topological defects famously follows a power law, $\xi\propto v^{\nu/(1+\nu z)}$, where $v$ is the velocity of the quench. By measuring the correlation length as a function of quench velocity, we can find the value of the exponent $\frac{\nu}{1+\nu z}$. This allow us to get equilibrium scaling exponents out of a non-equilibrium quench experiment!

We study a $\phi^4$ model with the following lagrangian:

$$\mathcal{L}=\frac{1}{2}\dot \phi -\frac{1}{2}(\nabla \phi)^2-V(\phi)$$

with 

$$V(\phi)=\frac{1}{8}(\phi^4-2\epsilon(t)\phi ^2)

$$

Noise and temperature require an outisde bath, which is modeled through a langevin extension to hte equation sof motion:

$$\ddot \phi + \eta \dot \phi - \nabla^2 \phi +V'(\phi)=\zeta(r,t)$$

Where $\zeta$ is our noise kernel satisfying:

$$\langle \zeta(r,t) \zeta(r',t') \rangle =2\eta \theta \delta^d(r-r')\delta(t-t')$$

The parameter of interest here is $\epsilon(t)$, which controls whether or not $V(\phi$) has one or two minima (In fact, $V(\phi)$ is exactly the mexican hat potential in the image above!)

We choose $\epsilon(t)=t/\tau$. We run an experiment from $t=-2\tau \to 10\tau$, which varies $\epsilon$ from -2 to 10 at a constant velocity $v=1/\tau$.


This picture is quite general. In this repository we consider the case of one and two dimension seperately. In two dimensions, we can also have coarsening dynamics, where regions of phase set by the kibble zurek dynamics, but then after this formation they can move, shrink, or grow. Oftentimes, anything that goes byeond mean-defect density is considered "beyond KZ physics". In this case, we aim to predict the entire field configuration, which is definitely beyond KZ physics!



# Results
### 1D replication study

We aim initially to replicate the results of [REFERENCE]. We simulate the dynamics of the system in one dimenison and train a Recurrent-Neural-Network to take in a snapshot of the field and predict the final field configuration.

<p align="center">
  <img src="figures/1d/figure1.png" width="700">
  <br>
  <em>Top: The real time evolution of the one dimensional field $\phi$ for a specific noise realization. Regions of positivity and negativity are evidence of the KZ dynamcis. These regions grow increasingly polarized as the minimum depth $\propto \sqrt{\epsilon}$ is increased. Bottom: The finial field configuration for this trajectory, with defects counted as zero-crossings of the field. The job of our RNN is to predict hte bottom plot given the top plot.</em>
</p>



You can train the model yourself, using `notebook.01_1d_replication.ipynb`. If you do, you'll find a result similar to this:


<p align="center">
  <img src="figures/1d/Predicted_Final_1d.png" width="700">
  <br>
  <em>The (normalized) field input in blue, the true final field in black, and our RNN's prediction of the final field in Red. As we can see, the model is not only capable of learning the final number of defects, but is also capable of predicting defect location!</em>
</p>






<p align="center">
  <img src="figures/1d/defect_location_prediction.png" width="700">
  <br>
  <em>Location of defects versus the prediction by the model</em>
</p>

In direct comparison with some of the figures in [Reference], we find that we replicate their results to a sufficient level of satisfaction. For instance, observe the training/validation curves we recover to the ones reported by them.

<p align="center">
  <img src="figures/1d/final_t_validation_curves.png" width="48%">
  <img src="figures/repo_fig/Paper_compare.png" width="48%">
  <br>
  <em>Left: Our result Right: [REFERENCE] result.</em>
</p>

#### Going Beyond the paper

Before continuing onto the two dimensional case, we first ask (as we will ask throughout this project): What is the model really learning? 

To help answer this question, we apply a low-pass filter to the inputs before asking our model to predict the final field values. KZ physics tells us that length scales less than $\xi_{KZ}$ (and hence, momentum modes larger than $k=\frac{1}{\xi_{KZ}}$) should be irrelevant to the final field configuration.

In applying a low pass filter, we are selecting to keep only momentum modes $k\in [0,k_c]$ where $k_c$ is a cutoff frequency.


We find that for $k_c\le \frac{1}{\xi_{KZ}}$, validation error greatly increases, but for $k_c\ge\frac{1}{\xi_{KZ}}$ is flat! The RNN is only learning features on the scale of the Kibble-Zurek length!


<p align="center">
  <img src="figures/1d/lowpass_training.png" width="700">
  <br>
  <em>Validation errors against $k_c$cutoff frequency. </em>
</p>

## 2D Kibble Zurek Physics

In two dimensions, there are different dynamical constraints at play in the evolution of the system. Now, defects form as two-dimensional areas of like-domain size.

This can be seen in the animation at the top of this page. We now want to ask the same question, where is the information in this quench? Does it change in two dimensions? Do effects like coarsening effect our dynamics? What does our model really end up learning?

To start, we examine a single image snapshot of the field $\phi$ at a single snapshot in time. We ask ourselves how much of the final field is already held in this image? 

Before using machine learning, it would be nice to establish a baseline of what we can assume about the dynamics from a snapshot. The most mild assumption we can make is that the field diffuses uniformly.





<p align="center">
  <img src="figures/2d/blur_widths_demo_tau32.png" width="700">
  <br>
  <em>A gaussian blur filter can reduce error in even a very disorderd state. </em>
</p>


We directly compare the error of the best-blur filter with the fitting results of our NN. 


<p align="center">
  <img src="figures/2d/unet_vs_blur_vs_persistence.png
" width="700">
  <br>
  <em>We compare the baseline error (blue) to the diffusive filter (orange) and our model (green). We see that the filter plateaus in its ability to predict the dynamics where the model is able to learn to predict the final distribution. At very disordered cases, the model can only learn the filter itself.  </em>
</p>


# Repository Structure

In `/src/` we have functions for generating the $\phi$ field ( `kz.py`, `kz_2d.py`) and for the subsequent analysis (`kz_ml.py`, `analysis.py`). Scripts for generating the fields in large batches suitable for machine learning purposes are found in `/scripts/kz1d/` and `/scripts/kz2d/` respectively. Each folder also has a respective `train.py` for impleneting a RNN or UNET respectively for either the 1D or 2D case. 

We provide a series of interactive Jupyter Notebook files that walk a reader through the experiments, and subsequent training and machine learning results.

# References 

1. F. Suzuki, Y. W. Li, and W. H. Zurek, *Machine learning topological defect formation: When are the defects made?*, Phys. Rev. Lett. (accepted), arXiv:2508.20347 (2026).


