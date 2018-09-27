---
layout: post
title: "Normalizing Flows: Planar and Radial Flows"
excerpt: A normalizing flow is a great tool that can transform simple probability distributions into very complex ones by applying a series of invertible functions to samples from the simple distribution. This post explores two simple flows introduced by Rezende et. al. –– Planar Flow and Radial Flow.
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

Simple distributions (e.g., Gaussian) are often used as likelihood distributions. However, the true distribution is often far from this simple distribution and this results in issues such as blurry reconstructions in the case of images. Latent variable models such as VAEs often set the prior distribution $p(\mathbf{z})$ to a factorial multivariate Gaussian distribution. Such a simplistic assumption hampers the model in multiple ways. For instance, this does not allow a multi-modal latent space distribution. Normalizing Flows allow transformation of samples from a simple distribution (subsequently denoted by $q_0$) to samples from a complex distribution by applying a series of invertible flows.

### Distribution of a Simple Transformation of a RV

Before jumping into normalizing flows, let's consider a simple univariate distribution $p(x)=2x$ with support $x\in[0,1]$. Define a function $y = f(x) = x^2$. Note that $f(x)$ is monotonically increasing in $[0,1]$. What is the PDF of the variable $y$?

We can compute $p(y)$ using the CDFs as follows.

$$
\begin{align}
F_{Y}(y) &= P(Y \leq y)\\
&= P(X^2 \leq y)\\
&= P(X \leq \sqrt{y})\\
&= F_{X}(\sqrt{y})\\
\end{align}
$$

Now, $p(y) = F_{Y}'(y) = \frac{dF_{X}(\sqrt{y})}{dy}$ where

$$
\begin{align}
F_{X}(\sqrt{y}) &= \int_{0}^{\sqrt{y}} p(x) dx\\
&=\left[2\frac{x^2}{2}\right]_{0}^{\sqrt{y}}\\
&=y
\end{align}
$$

and finally differentiating w.r.t. $y$ we get $\frac{d(y)}{dy} = 1$ which means that $p(y) = \mathcal{U}(0,1)$.

### Change of Variables

The method described above can be extended to multivariate distributions $q_0(\mathbf{z})$ and smooth invertible mappings $f: \mathbb{R}^d\Rightarrow\mathbb{R}^d$. Samples $\mathbf{z} \sim q_0(\mathbf{z})$ can be transformed using $f$ to give $\mathbf{y}=f(\mathbf{z})$. The PDF of $\mathbf{y}$ is given by 

$$
q_1(\mathbf{y}) = q_0(\mathbf{z})\left|\det \frac{\partial f^{-1}}{\partial \mathbf{y}}\right| = q_0(\mathbf{z})\left|\det \frac{\partial f}{\partial \mathbf{z}}\right|^{-1}
$$
where the second equality comes from the inverse-function theorem.

Rezende et. al. proposed two different families of invertible transformations: planar flow and radial flow.

## Planar Flow

Planar flows use functions of form

$$
\begin{align*}
f(\mathbf{z}) = \mathbf{z} + \mathbf{u}h(\mathbf{w}^\top\mathbf{z} + b)
\end{align*}
$$

where $\mathbf{u},\mathbf{w}\in \mathbb{R}^d$, $b \in \mathbb{R}$, and $h$ is an element-wise non-linearity such as $\tanh$.

The Jacobian is defined as follows.

$$
\begin{align*}
    \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} = \mathbf{I} + \mathbf{u}h'(\mathbf{w}^\top\mathbf{z} + b)\mathbf{w}^\top
\end{align*}
$$

Now, using the matrix determinant lemma

$$
\begin{align*}
\det\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} &= (1 + h'(\mathbf{w}^\top\mathbf{z} + b)\mathbf{w}^\top\mathbf{I}^{-1}\mathbf{u})\det(\mathbf{I})\\
&=(1 + h'(\mathbf{w}^\top\mathbf{z} + b)\mathbf{w}^\top\mathbf{u})
\end{align*}
$$

(...to be continued)

