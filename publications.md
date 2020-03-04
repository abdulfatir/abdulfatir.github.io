---
layout: page
title: Publications
permalink: /publications/
---


<div class="container">
   <h2>2020</h2>
   <div class="row">
      <div class="col-md-12">
         <span class="text-info">A Characteristic Function Approach to Deep Implicit Generative Modeling</span>
         <br />
         <b>Abdul Fatir Ansari</b>, Jonathan Scarlett, Harold Soh.
         <br />
         <i>Conference on Computer Vision and Pattern Recognition (CVPR) 2020</i>
         <br />
         <a class="btn btn-outline-success btn-sm" data-toggle="collapse" href="#cvpr20abstract" role="button" aria-expanded="false" aria-controls="cvpr20abstract">Abstract</a>
         <a href="https://arxiv.org/abs/1909.07425" role="button" class="btn btn-outline-success btn-sm">Preprint</a>
         <a href="#" role="button" class="btn btn-outline-success btn-sm disabled" aria-disabled="true">Code (coming soon)</a>
         <div class="collapse" id="cvpr20abstract" style="padding: 5px;">
            <div class="card border-success mb-3">
               <div class="card-header">Abstract</div>
               <div class="card-body">
                  <div class="card-text text-justify font-italic" style="font-size:0.8rem;">
                  Implicit Generative Models (IGMs) such as GANs have emerged as effective data-driven models for generating samples, particularly images. In this paper, we formulate the problem of learning an IGM as minimizing the expected distance between characteristic functions. Specifically, we match the characteristic functions of the real and generated data distributions under a suitably-chosen weighting distribution. This distance measure, which we term as the characteristic function distance (CFD), can be (approximately) computed with linear time-complexity in the number of samples, compared to the quadratic-time Maximum Mean Discrepancy (MMD). By replacing the discrepancy measure in the critic of a GAN with the CFD, we obtain a model that is simple to implement and stable to train; the proposed metric enjoys desirable theoretical properties including continuity and differentiability with respect to generator parameters, and continuity in the weak topology. We further propose a variation of the CFD in which the weighting distribution parameters are also optimized during training; this obviates the need for manual tuning and leads to an improvement in test power relative to CFD. Experiments show that our proposed method outperforms WGAN and MMD-GAN variants on a variety of unsupervised image generation benchmarks.
                  </div>
               </div>
            </div>
         </div>
      </div>
   </div>
</div>

<div class="container">
   <h2>2019</h2>
   <div class="row">
      <div class="col-md-12">
         <span class="text-info">Hyperprior Induced Unsupervised Disentanglement of Latent Representations</span>
         <br />
         <b>Abdul Fatir Ansari</b> and Harold Soh.
         <br />
         <i>AAAI Conference on Artificial Intelligence (AAAI) 2019</i>
         <br />
         <a class="btn btn-outline-success btn-sm" data-toggle="collapse" href="#aaai19abstract" role="button" aria-expanded="false" aria-controls="aaai19abstract">Abstract</a>
         <a href="https://arxiv.org/abs/1809.04497" role="button" class="btn btn-outline-success btn-sm">Paper</a> <a href="https://github.com/crslab/CHyVAE" role="button" class="btn btn-outline-success btn-sm">Code</a> <a href="https://github.com/crslab/correlated-ellipses" role="button" class="btn btn-outline-success btn-sm">Dataset</a>
         <a class="btn btn-outline-success btn-sm" href="{{ site.base }}/files/bib/aaai19cite.txt" target="_blank" role="button">Cite</a>
         <div class="collapse" id="aaai19abstract" style="padding: 5px;">
            <div class="card border-success mb-3">
               <div class="card-header">Abstract</div>
               <div class="card-body">
                  <div class="card-text text-justify font-italic" style="font-size:0.8rem;">
                  We address the problem of unsupervised disentanglement of latent representations learnt via deep generative models. In contrast to current approaches that operate on the evidence lower bound (ELBO), we argue that statistical independence in the latent space of VAEs can be enforced in a principled hierarchical Bayesian manner. To this effect, we augment the standard VAE with an inverse-Wishart (IW) prior on the covariance matrix of the latent code. By tuning the IW parameters, we are able to encourage (or discourage) independence in the learnt latent dimensions. Extensive experimental results on a range of datasets (2DShapes, 3DChairs, 3DFaces and CelebA) show our approach to outperform the β-VAE and is competitive with the state-of-the-art FactorVAE. Our approach achieves significantly better disentanglement and reconstruction on a new dataset (CorrelatedEllipses) which introduces correlations between the factors of variation.
                  </div>
               </div>
            </div>
         </div>
      </div>
   </div>
</div>
