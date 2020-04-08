---
layout: post
title: "1-Wasserstein distance: Kantorovich–Rubinstein duality"
excerpt: The 1-Wasserstein distance is a popular integral probability metric. In this post, the dual form of the 1-Wasserstein distance is derived from its primal form.
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

The Kantorovich–Rubinstein distance, popularly known to the machine learning community as the Wasserstein distance, is a metric to compute the distance between two probability measures. The 1-Wasserstein is the most common variant of the Wasserstein distances (thanks to WGAN and its variants). Its dual form is generally used in an adversarial setup which is defined as

$$
\sup_{f} \int fd\mu(x) - \int fd\nu(y) \quad\text{  where  } f:\mathbb{R}^d\rightarrow\mathbb{R},\:\mathrm{Lip}(f) \leq 1
\tag{1}\label{eq:w1dual}
$$

where $\mu$ and $\nu$ are probability measures, and $\mathrm{Lip}(f)$ denotes the Lipschitz constant of the function $f$. The function $f$ is realized using a neural network and the Lipschitz constraint is enforced using various techniques such as weight-clipping, gradient penalty, and spectral normalization. The neural network is trained to maximize the integral $\int fd\mu(x) - \int fd\nu(y)$.

Eq. ($\ref{eq:w1dual}$) is the dual form of the Wasserstein distance which has the general form

$$
W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int d(x,y)^pd\gamma(x,y)\right)^{1/p}
\tag{2}\label{eq:wpprimal}
$$

where $d(x,y)$ is a distance metric between two points $x$ and $y$, and $\gamma$ is a coupling of the probability measures $\mu$ and $\nu$. Without going into too much jargon, a coupling $\gamma$ can be thought of as a joint distribution that belongs to the set of joint distributions $\Gamma(\mu, \nu)$ that have marginals $\mu$ and $\nu$. For the remainder of this note, we'll set $d(x,y) = \|x-y\|$ and $p=1$.

Generally in machine learning literature, the dual form and the Lipschitz constraints are directly introduced and there's a lack of resources on the internet that address these basics. To this end, in this post I will explain how to arrive at the dual form (Eq. $\ref{eq:w1dual}$) from the primal form: 

$$
W_1(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x-y\|d\gamma(x,y).
\tag{3}\label{eq:w1primal}
$$

The treatment in this post will be a non-rigorous one. Please refer to [1] for a rigorous proof.

## Kantorovich Duality

We first remove the $\gamma \in \Gamma(\mu, \nu)$ constraint from Eq. ($\ref{eq:w1primal}$) and add it as a supremum instead.

$$
W_1(\mu, \nu) = \inf_{\gamma} \int \|x-y\|d\gamma(x,y) + \underset{\gamma \in \Gamma(\mu, \nu)\:\text{constraint}}{\underbrace{\sup_{f,g} \int fd\mu(x) + \int gd\nu(y) - \int (f(x)+g(y))d\gamma(x,y)}}
\tag{4}\label{eq:derivstep1}
$$

where $f$ and $g$ are absolutely integrable functions w.r.t. $\mu$ and $\nu$ respectively. Eq. ($\ref{eq:derivstep1}$) encodes the $\gamma \in \Gamma(\mu, \nu)$ constraint because the supremum is $0$ when $\gamma \in \Gamma(\mu, \nu)$ and is $\infty$ otherwise. This is because 

$$
\int (f(x)+g(y))d\gamma(x,y) = \int f(x)d\mu(x) + \int g(y)d\nu(y)\quad\text{if } \gamma \in \Gamma(\mu, \nu)
$$

and would cancel the other terms. In any other case, $f$ and $g$ can be suitably chosen such that the supremum becomes $\infty$.

We can now take the $\sup_{f,g}$ outside because the first term does not depend on $f$ and $g$,

$$
W_1(\mu, \nu) = \inf_{\gamma}\sup_{f,g} \int (\|x-y\|-f(x)+g(y))d\gamma(x,y) + \int fd\mu(x) + \int gd\nu(y)
$$

We can now invoke the minimax principle to replace the $\inf\sup$ with a $\sup\inf$ under certain conditions which are beyond the scope of this post.

$$
W_1(\mu, \nu) = \sup_{f,g}\inf_{\gamma} \int (\|x-y\|-f(x)+g(y))d\gamma(x,y) + \int fd\mu(x) + \int gd\nu(y)
$$

When $\|x-y\| \geq f(x)+g(y)$, the value of $\inf_{\gamma} \int (\|x-y\|-f(x)+g(y))d\gamma(x,y)$ is $0$ and is $-\infty$ otherwise. This can be added as a constraint in the equation as follows

$$
W_1(\mu, \nu) = \sup_{\substack{f,g\\f(x)+g(y) \leq \|x-y\|}} \int fd\mu(x) + \int gd\nu(y)
\tag{5}\label{eq:w1dualgeneral}
$$

### $c$-transform

How do we now find such $f$ and $g$ the maximize the r.h.s. of Eq. ($\ref{eq:w1dualgeneral}$)?

Let's assume that we have a function $f$ and we want to find the optimal $g$ corresponding to $f$ that achieves the supremum in Eq. ($\ref{eq:w1dualgeneral}$). We know that $\forall x,y \, f(x) + g(y) \leq \|x-y\|$. We can write this as follows

$$
g(y) \leq \inf_{x} \|x-y\| - f(x)
$$

because if $g(y) \leq \|x-y\| - f(x) \, \forall x,y$, then it must be true for the $x$ that minimizes the r.h.s. (which also makes sure that it is true for all other $x$s). Since, $g(y) \leq \inf_{x} \|x-y\| - f(x)$, the best we can do to maximize the r.h.s. in Eq. ($\ref{eq:w1dualgeneral}$) is set 

$$
g(y) = \inf_{x} \|x-y\| - f(x)
\tag{6}\label{eq:ctransform1}
$$.

Eq. ($\ref{eq:ctransform1}$) gives us a function which is called the $c$-transform of $f$ and is often denoted by $f^c$,

$$
f^c(y) = g(y) = \inf_{x} \|x-y\| - f(x)
$$

It can be shown that $f^{cc} = f$. We can now write Eq. ($\ref{eq:w1dualgeneral}$) as

$$
W_1(\mu, \nu) = \sup_{f} \int fd\mu(x) + \int f^cd\nu(y)
\tag{7}\label{eq:w1dualffc}
$$

### Special case for cost $\| x-y \|$

The above duality holds for any arbitrary cost $c(x,y)$, not just for $c(x,y)=\|x-y\|$. In the case of $\|x-y\|$, let's derive the form of the dual problem (Eq. $\ref{eq:w1dualgeneral}$) when $f$ is 1-Lipschitz.

When $f$ is 1-Lipschitz, $f^c$ is 1-Lipschitz too. This is true because for any given $x$,

$$f^c(y;x) = \|x-y\| - f(x)$$

is 1-Lipschitz and therefore the infimum of the r.h.s. $f^c(y) = \inf_{x} \|x-y\| - f(x)$ is 1-Lipschitz.

Since $f^c$ is 1-Lipschitz, for all $x$ and $y$ we have

$$
\begin{align}
&|f^c(x) - f^c(y)| \leq 1\cdot\|x-y\|\\
&\implies -1\cdot\|x-y\| \leq f^c(x) - f^c(y) \leq 1\cdot\|x-y\|\\
&\implies - f^c(x) \leq \|x-y\| - f^c(y)\tag{8}\label{ineq:lip}
\end{align}
$$

Since Eq. ($\ref{ineq:lip}$) is true for all $x$ and $y$,

$$
\begin{align}
&\implies-f^c(x) \leq \inf_{y} \|x-y\| - f^c(y)\\
&\implies-f^c(x) \leq \underset{f^{cc}(x)}{\underbrace{\inf_{y} \|x-y\| - f^c(y)}} \leq -f^c(x)\tag{9}\label{ineq:sandwich}
\end{align}
$$

where the right inequality follows by choosing $y = x$ in the infimum. We know that $f^{cc} = f$. This means that $-f^c(x)$ must be equal to $f(x)$ for Eq. ($\ref{ineq:sandwich}$) to hold. 

Substituting $f^c = -f$ in Eq. ($\ref{eq:w1dualffc}$), we get 

$$
\sup_{\substack{f\\\mathrm{Lip}(f) \leq 1}} \int fd\mu(x) - \int fd\nu(y)
$$

which is the dual form of 1-Wasserstein distance.

## References

[1] Cédric Villani: Topics in Optimal Transportation, Chapter 1.    
[2] Vincent Herrmann : Wasserstein GAN and the Kantorovich-Rubinstein Duality ([Blog](https://vincentherrmann.github.io/blog/wasserstein/)).    
[3] Marco Cuturi: A Primer on Optimal Transport ([Talk](https://www.youtube.com/watch?v=1ZiP_7kmIoc)).    
