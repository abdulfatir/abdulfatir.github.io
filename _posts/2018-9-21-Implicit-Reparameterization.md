---
layout: post
title: Implicit Reparameterization Gradients
---

Backpropagation through a stochastic node is an important problem in machine learning. The optimization of $$\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$ requires computation of $$\nabla_\phi\mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$$.
