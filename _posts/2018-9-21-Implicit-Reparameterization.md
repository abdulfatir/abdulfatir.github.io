---
layout: post
title: Implicit Reparameterization Gradients
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ TeX: { extensions: ["color.js"] }});
</script>

Backpropagation through a stochastic node is an important problem in deep learning. The optimization of $$\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$ requires computation of $$\nabla_\phi\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$. Stochastic variational inference requires the computation of the gradient of one such expectation.

$$
\definecolor{mBrown}{RGB}{188,99,16}
$$
$$
\begin{align*}
    \mathcal{L}(\mathbf{x},\theta, \phi) = \color{mBrown}{\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]} - \color{black}\mathrm{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))
\end{align*}
$$

Earlier methods of gradient computation include score-function-based estimators (REINFORCE) and pathwise gradient estimators (reparameterization trick). Recent works have proposed using reparametrizable surrograte distributions such as $$\mathrm{GumbelSoftmax}$$ for $$\mathrm{Categorical}$$, $$\mathrm{Kumaraswamy}$$ for $$\mathrm{Beta}$$, etc. Other recent works such as Generalized Reparameterization Gradients (GRG) and Rejection Sampling Variational Inference (RSVI) have sought to build a generalized framework for gradient computation.
