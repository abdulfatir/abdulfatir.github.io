---
layout: post
title: Implicit Reparameterization Gradients
---

Backpropagation through a stochastic node is an important problem in deep learning. The optimization of $$\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$ requires computation of $$\nabla_\phi\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$. Stochastic variational inference requires the computation of the gradient of one such expectation.

$$
\begin{align*}
    \mathcal{L}(\mathbf{x},\theta, \phi) = \textcolor{red}{\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]} - \mathrm{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))
\end{align*}
$$
