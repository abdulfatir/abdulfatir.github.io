---
layout: post
title: "Introduction to Gradient Flows in the 2-Wasserstein Space"
date: 2020-12-24 21:01:00
disqus_comments: true
featured: true
tags: optimal-transport gradient-flows
description: Gradient flows have been a popular tool in the analysis of PDEs. Recently, various gradient flows have been studied in machine learning literature. This article is an introduction to the concept of gradient flows in the 2-Wasserstein space.
---

------------------
<span style="font-size: 80%; color: gray;">
**Note**: This article is a gentle introduction to the concept of gradient flows in the Wasserstein space. The target audience are researchers working in the area of machine learning, not mathematicians (forgive my non-rigorous soul). The "flow" of arguments in this article follows the excellent [overview of gradient flows](https://arxiv.org/abs/1609.03890) by [Filippo Santambrogio](http://math.univ-lyon1.fr/~santambrogio/) and [this lecture](https://www.youtube.com/watch?v=zzGBxAqJV0Q) by [Brittany Hamfeldt](https://web.njit.edu/~bdfroese/).
</span>

------------------

$$
\definecolor{purple}{RGB}{114,0,172}
\definecolor{maroon}{RGB}{133, 20, 75}
\definecolor{blue}{RGB}{18,110,213}
$$

A gradient flow is a curve following the direction of steepest descent of a function(-al). For example, let $E: \mathbb{R}^n \to \mathbb{R}$ be a smooth, convex energy function. The gradient flow of $E$ is the solution to the following initial value problem,

$$
\begin{align}
x'(t) &= -\nabla E(x(t))\tag{1}\label{eq:euclidean-gf},\\
x(0) &= x_0.
\end{align}
$$

We seek to extend the idea of steepest descent curves to metric spaces, specifically the 2-Wasserstein space (defined later); however, tangent vectors and gradients have no definitions in metric spaces. In the following, we will characterize the gradient flow in Eq. ($\ref{eq:euclidean-gf}$) as the limit curve of a discrete-time scheme. We begin by discretizing the Ordinary Differential Equation (ODE) using the implicit Euler method,

$$
\begin{align}
\frac{x_{n+1} - x_{n}}{\tau} &= -\nabla E(x_{n+1})\tag{2}\label{eq:back-euler},
\end{align}
$$

which can equivalently be written as 

$$
\begin{align}
\nabla\left(\frac{|x - x_{n}|^2}{2\tau} + E(x)\right)\Bigg|_{x=x_{n+1}} = 0\tag{3}\label{eq:back-euler-optim}.
\end{align}
$$

Eq. ($\ref{eq:back-euler-optim}$) looks like an optimality condition and consequently $x_{n+1}$ can be written as the solution to a minimization problem,

$$
\begin{align}
x_{n+1} = \mathrm{arg}\min\left(\frac{|x - x_{n}|^2}{2\tau} + E(x)\right)\tag{4}\label{eq:mms-gf}.
\end{align}
$$

We have converted the discrete scheme into a form that does not involve gradients. This discrete scheme can be generalized to a metric space $(\mathcal{X}, d)$ where the space $\mathcal{X}$ is endowed with the metric $d$. Let $F: \mathcal{X} \to \mathbb{R}$ be a functional that is lower semi continuous and bounded below<span style="font-size: 70%; color: gray;"><sup>1</sup></span>. The equivalent discrete scheme in this metric space is given by,

$$
\begin{align}
x_{n+1}^\tau \in \mathrm{arg}\min\left(\frac{d(x, x_{n}^\tau)^2}{2\tau} + F(x)\right)\tag{5}\label{eq:gmm-gf},
\end{align}
$$

where we have replaced the Euclidean metric with the metric $d$. We can define a piecewise constant, continuous-time interpolation from the discrete scheme in Eq. ($\ref{eq:gmm-gf}$) as follows

$$
x^{\tau}(t) = x_n^\tau\qquad \text{if}\:\: t \in ((n-1)\tau, n\tau]\tag{6}\label{eq:pc-interpolation}.
$$

Gradient flows can now be characterized by studying the discrete scheme in the limit $\tau \to 0$. This discrete scheme is called the minimizing movement scheme and a curve $x: [0, T] \to \mathcal{X}$ is called Generalized Minimizing Movements (GMM) if there exists a sequence of time steps $\tau_j \to 0$ such that the sequence of piecewise constant curves in Eq. (\ref{eq:pc-interpolation}) converges uniformly to $x$. 

Before discussing the 2-Wasserstein space, we will define some quantities in metric spaces that will be referred to later.

**Metric Derivative** For a curve $x: [0, T] \to \mathcal{X}$ valued in a metric space $\mathcal{X}$, we can define the modulus of $x'(t)$ (i.e., the speed of the curve instead of its velocity as one would do in a vector space) as 

$$
|x'|(t) \triangleq \lim_{h \to 0} \frac{d(x(t), x(t + h))}{|h|}.
$$

**AC Curves in a Metric Space** A curve $x: [0, T] \to \mathcal{X}$ is said to be absolutely continuous (AC) if there exists a $g \in L^1([0, 1])$ such that $d(x(t_0), x(t_1)) \leq \int_{t_0}^{t_1}g(s)ds$ for every $t_0 < t_1$.

## Background in Optimal Transport

Before defining the Wasserstein space of probability measures we introduce some ideas from optimal transport that will be used later. The optimal transport (OT) problem seeks to transport mass from a source measure $\mu$ to a target measure $\nu$ with respect to a given cost function $c$ in an "optimal" fashion. There are two popular formulations of the optimal transport problem due to Monge and Kantorovich.

**Monge's Formulation** Monge's formulation seeks to find an (optimal) transport map $T$ that pushes forward $\mu$ to $\nu$ (often denoted as $T_{\sharp}\mu = \nu$)<span style="font-size: 70%; color: gray;"><sup>2</sup></span> while minimizing the following quantity

$$
\inf_{T_{\sharp}\mu = \nu} \int_{\mathcal{X}} c(T(x), x)d\mu(x)\tag{MP}.
$$

Note that we're looking for a transport map and are not allowed to "break up mass". Consider the problem of transporting a dirac measure at $0$ to a measure that assigns equal mass to $-1$ and $+1$. It is not possible to construct a Monge map for such a transport problem. 

**Kantorovich's Formulation** Kanotrovich's formulation seeks an (optimal) transport plan $\gamma$ that transports mass from $\mu$ to $\nu$ with respect to the cost function $c$ such that the marginals of $\gamma$ are $\mu$ and $\nu$.

$$
\inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y)d\gamma(x,y)\tag{KP}\label{eq:ot-kp}
$$

where $\Gamma(\mu, \nu)$ is the set of transport plans with the correct marginals, i.e., 

$$
\Gamma(\mu, \nu) = \{\gamma | (\pi_{\mathcal{X}})_\sharp\gamma = \mu, (\pi_{\mathcal{Y}})_\sharp\gamma=\nu\}
$$

where $$(\pi_{\mathcal{X}})_\sharp\gamma$$ denotes the projection of $\gamma$ onto $$\mathcal{X}$$. Monge's formulation is a special case of Kanotrovich's formulation such that whenever the optimal Monge map $$T^*$$ exists, the corresponding Kantorovich transport plan is given by $$\gamma^* = (\mathrm{id} \times T^*)_\sharp\mu$$ where $$\mathrm{id}$$ is the identity map.

**Kantorovich duality** Kanotrovich's formulation has an equivalent dual form given by 

$$
\sup_{\substack{f,g\\f(x)+g(y) \leq c(x, y)}} \int_\mathcal{X} fd\mu(x) + \int_\mathcal{Y} gd\nu(y)\tag{KPDual}\label{eq:ot-kpdual},
$$

where the supremum runs over the set of bounded and continuous functions $f: \mathcal{X} \to \mathbb{R}$ and $g: \mathcal{Y} \to \mathbb{R}$ such that $$f(x)+g(y) \leq c(x, y)$$. Eq. (\ref{eq:ot-kpdual}) can be written as a supremum over a single function $\psi$ with its corresponding $c$-transform<span style="font-size: 70%; color: gray;"><sup>3</sup></span> $\psi^c$,

$$
\sup_{\psi \in \Psi^c} \int_\mathcal{X} \psi d\mu(x) + \int_\mathcal{Y} \psi^c d\nu(y)\tag{7}\label{eq:ot-kpdual-ctransform},
$$

where $$\psi^c(y) = \inf_{x} \left\{c(x, y) - \psi(x)\right\}$$ and $\Psi^c$ is the set of $c$-concave functions where a $c$-concave function is a function $\psi$ for which there exists a function $\varphi$ such that $\psi = \varphi^c$. The function(s) $\psi$ realizing the maximum in Eq. (\ref{eq:ot-kpdual-ctransform}) are called Kantorovich potentials.

**OT with Quadratic Cost** For the special case of optimal transport with the quadratic cost $$c(x, y) = \frac{\|x - y\|^2}{2}$$, there exists a unique optimal transport plan of the form $(\mathrm{id} \times T^*)_\sharp\mu$ provided that $\mu$ is absolutely continuous. Further, there exists at least one Kantorovich potential $\psi$ such that its gradient $\nabla \psi$ is uniquely determined. The form of the transport plan implies the existence of an optimal Monge map $$T^*$$ which is related to the potential $\psi$ by $$T^*(x) = x - \nabla\psi(x)$$.

## The 2-Wasserstein Space

Let $\mathcal{P}_2(\Omega)$ be the set of probability measures on a domain $\Omega \subset \mathbb{R}^d$ with finite second moments, i.e.,

$$
\mathcal{P}_2(\Omega) \triangleq \left\{\mu \Big| \int_{\Omega}|x|^2d\mu(x) < \infty\right\}.
$$

The 2-Wasserstein distance between probability measures $\mu \in \mathcal{P}_2(\Omega)$ and $\nu \in \mathcal{P}_2(\Omega)$ is defined as, 

$$

{\color{blue}\mathcal{W}_2(\mu, \nu) \triangleq \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x-y\|^2d\gamma(x,y)\right)^{1/2}},
\tag{W2}\label{eq:w2primal}
$$

where $\Gamma(\mu, \nu)$ is a set of all possible couplings with marginals $\mu$ and $\nu$. We can see that $\mathcal{W}_2$ is equal to the square root of Eq. (\ref{eq:ot-kp}) with $$c(x,y) = \|x-y\|^2$$. It can be shown that $\mathcal{W}_2$ satisfies the axioms of a metric on $\mathcal{P}_2(\Omega)$ and convergence with respect to $\mathcal{W}_2$ is equivalent to weak convergence of probability measures, i.e., for a sequence of measures $$(\mu_k)_{k \in \mathbb{N}}$$ in $\mathcal{P}_2(\Omega)$, $$\mu_k \to \mu \Longleftrightarrow \mathcal{W}_2(\mu_k, \mu) \to 0$$. The 2-Wasserstein space $(\mathcal{P}_2(\Omega), \mathcal{W}_2)$ is the metric space of probability measures $\mathcal{P}_2(\Omega)$ endowed with the 2-Wasserstein ($\mathcal{W}_2$) metric. 

**AC Curves in the 2-Wasserstein Space** Let $$\{\mu_t\}_{t\in[0,1]}$$ be an absolutely continuous curve in $(\mathcal{P}_2(\Omega), \mathcal{W}_2)$. Then, for $t \in [0, 1]$ there exists a velocity field $\mathbf{v}_t \in L^2(\mu_t; \mathbb{R}^d)$ such that the continuity equation

$$
{\color{blue}\frac{\partial \mu_t}{\partial t} + \nabla\cdot(\mathbf{v}_t\mu_t) = 0}\tag{CE}\label{eq:continuity-eq}
$$

is satisfied and 
 
$$
\|\mathbf{v}_t\|_{L^2(\mu_t)} = |\mu'|(t).\tag{8}\label{eq:velocity-norm-metric-derivative}
$$

The proof of these statements is beyond the scope of this article (frankly, I don't know how to prove it). However, for the second statement, we can consider two measures $\mu_t$ and $\mu_{t+h}$. There are several ways to transport mass from $\mu_t$ to $\mu_{t+h}$ one of which is optimal in the OT sense. Let $T^*$ be the OT map between $\mu_t$ and $\mu_{t+h}$. We can define $$\mathbf{v}_t(x)$$ as the discrete velocity of the particle $x$ at time $t$ given by $$\mathbf{v}_t(x) = (T^*(x)-x)/h$$ (i.e., displacement/time). We can intuitively see that in the limit $$h \to 0$$, Eq. (\ref{eq:velocity-norm-metric-derivative}) holds, since

$$
\begin{align}
\|\mathbf{v}_t\|_{L^2(\mu_t)} &= \left(\int_{\mathbb{R}^d}\left|\frac{(T^*(x)-x)}{h}\right|^2d\mu_t(x)\right)^{1/2},\\
&= \frac{1}{h}\mathcal{W}_2(\mu_{t},\mu_{t+h}).
\end{align}
$$

## Gradient Flows in the 2-Wasserstein Space

Now that we have established that absolutely continuous curves in the 2-Wasserstein space satisfy the continuity equation, our task is to link Partial Differential Equations (PDEs) of the form of Eq. (\ref{eq:continuity-eq}) to the discrete-time minimizing movement scheme,

$$
\begin{align}
\rho_{n+1}^\tau &\in \mathrm{arg}\min\left(\frac{\mathcal{W}_2(\rho, \rho_{n}^\tau)^2}{2\tau} + \mathcal{F}(\rho)\right)\tag{9}\label{eq:mms-w2},\\
\rho^\tau(t) &= \rho^\tau_n\qquad \text{if}\:\: t \in ((n-1)\tau, n\tau],
\end{align}
$$

where $\mathcal{F}: \mathcal{P}_2(\Omega) \to \mathbb{R}$ is a functional on the 2-Wasserstein space that is lower semi-continuous and bounded below. Note that we now denote probability measures using $\rho$ to indicate the fact that they are absolutely continuous measures with smooth densities. Concretely, now our task is to find the velocity field $\mathbf{v}_t$ such that the solution to the continuity equation agrees with $$\lim_{\tau\to 0}\rho^\tau(t)$$.

Let us now investigate the optimality condition of Eq. (\ref{eq:mms-w2}). By analogy to the optimality condition for functions where we set the first derivative of the function equal to 0, we can set the first variation of a functional defined on the 2-Wasserstein space equal to a constant (Why?<span style="font-size: 70%; color: gray;"><sup>2</sup></span>). The first variation $${\color{purple}\frac{\delta\mathcal{G}}{\delta\rho}}$$ of a functional $$\mathcal{G}: \mathcal{P}_2(\Omega) \to \mathbb{R}$$ at a point $$\rho$$, if it exists, is defined (up to additive constants) as 

$$
\frac{d}{d\epsilon} \mathcal{G}(\rho + \epsilon\chi)\Bigg|_{\epsilon = 0} = \int {\color{purple}\frac{\delta\mathcal{G}}{\delta\rho}(\rho)}d\chi(x).
$$

Note that $\chi$ is chosen such that $\rho + \epsilon\chi \in \mathcal{P}_2(\Omega)$; this can be done by setting $\chi = \sigma - \rho$ for some $\sigma \in \mathcal{P}_2(\Omega)$.

We now compute the first variation of the functional on the r.h.s. of Eq. (\ref{eq:mms-w2}) and set it equal to a constant,

$$
\begin{align}
\frac{\delta}{\delta \rho}\left[\frac{\mathcal{W}_2(\rho, \rho_{n}^\tau)^2}{2\tau} + \mathcal{F}(\rho)\right]\Bigg|_{\rho=\rho^\tau_{n+1}} &= \mathrm{constant},\\
\left[\frac{1}{2\tau}\frac{\delta\mathcal{W}_2(\rho, \rho_{n}^\tau)^2}{\delta \rho} + \frac{\delta\mathcal{F}}{\delta \rho}(\rho)\right]\Bigg|_{\rho=\rho^\tau_{n+1}} &= \mathrm{constant}.\tag{10}\label{eq:optimality-eq}\\
\end{align}
$$

The first variation of $\mathcal{F}$ depends on its specific form; therefore, we begin by deriving an expression for the first variation of the squared 2-Wasserstein distance.

**First Variation of the Squared 2-Wasserstein Distance** We begin by writing the expression of the squared Wasserstein distance in its primal form,

$$
\mathcal{W}^2_2(\mu, \nu) = 2\color{maroon}{\inf_{\gamma \in \Gamma(\mu, \nu)} \int \frac{\|x-y\|^2}{2}d\gamma(x,y)}.
$$

The expression in <span style="color:#85144b">maroon</span> is the Kantorovich formulation of OT with the quadratic cost $$c(x,y) = \frac{\|x-y\|^2}{2}$$. We have multiplied and divided the expression by 2 to utilize the properties of OT with the quadratic cost. Let us convert the expression into its dual form using $c$-concave functions,

$$
\mathcal{W}^2_2(\mu, \nu) = 2{\sup_{\psi \in \Psi^c} \int \psi d\mu(x) + \int \psi^c d\nu(y)}.
$$

Let $\psi_*$ be a Kantorovich potential that achieves the supremum in the above equation. We can now replace $$\psi$$ with $$\psi_*$$ and remove the $$\sup$$ operator,

$$
\mathcal{W}^2_2(\mu, \nu) = 2 \int \psi_* d\mu(x) + \int \psi_*^c d\nu(y).
$$

Perturbing $\mu$ along $\chi$ and differentiating the resulting expression with respect to $\epsilon$ at $\epsilon=0$ we get,

$$
\frac{d}{d\epsilon} \mathcal{W}^2_2(\mu + \epsilon\chi, \nu)\Bigg|_{\epsilon = 0} = \int {\color{purple}\underset{\frac{\delta\mathcal{W}^2_2(\mu, \nu)}{\delta\mu}}{\underbrace{2\psi_*}}} d\chi(x).
$$

**Deriving an Expression for Particle Velocity** The above equation shows that the first variation of the squared Wasserstein distance between measures $\mu$ and $\nu$ is equal to $$2\psi_*$$ where $$\psi_*$$ is the Kantorovich potential associated with optimal transport from $\mu$ to $\nu$ with the quadratic cost. We can substitute this result into Eq. (\ref{eq:optimality-eq}) and get,

$$
\left[\frac{\varphi_*}{\tau} + \frac{\delta\mathcal{F}}{\delta \rho}(\rho^\tau_{n+1})\right] = \mathrm{constant}\tag{11}\label{eq:optimality-cond-2},
$$

where $\varphi_*$ is the Kantorovich potential associated with optimal transport from $\rho^\tau_{n+1}$ to $\rho^\tau_{n}$. As mentioned earlier, the Kantorovich potential and the Monge transport map for the quadratic cost are related by $$T^*(x) = x - \nabla\varphi_*(x)$$. Taking the gradient with respect to $x$ on both sides of Eq. (\ref{eq:optimality-cond-2}) and substituting $$\nabla\varphi_*(x) = -(T^*(x)-x)$$ we get

$$
\frac{(T^*(x)-x)}{\tau} = \nabla\frac{\delta\mathcal{F}}{\delta \rho}(\rho^\tau_{n+1})(x).
$$

Note that the expression $$\frac{(T^*(x)-x)}{\tau}$$ can be thought of as the discrete velocity of a particle moving backwards in time from $\rho^\tau_{n+1}$ to $\rho^\tau_{n}$ in an optimal transport sense. We can intuitively see how this expression would become the instantaneous velocity in the limit $\tau \to 0$. This gives us an expression for the instantaneous forward velocity of a particle, 

$$
{\color{blue}\mathbf{v}_t(x) = -\nabla\frac{\delta\mathcal{F}}{\delta \rho}(\rho_t)(x)}.
$$

Plugging this expression into the continuity equation we get the expression of the gradient flow of a functional $\mathcal{F}$ in the Wasserstein space

$$
{\color{blue}\frac{\partial \rho_t}{\partial t} - \nabla\cdot\left(\nabla\frac{\delta\mathcal{F}}{\delta \rho}(\rho_t)\rho_t\right) = 0}
$$

### Some famous examples of gradient flows in $(\mathcal{P}_2(\Omega), \mathcal{W}_2)$

**Negative Differential Entropy** When the functional $\mathcal{F}$ is defined to be the negative differential entropy, i.e.,

$$
\mathcal{F} \triangleq \int \rho(x)\log\rho(x)dx,
$$

the velocity is given by $$\mathbf{v}_t = -\frac{1}{\rho(x)}\nabla\rho(x)$$ and we recover the famous heat equation as the gradient flow of negative entropy in the 2-Wasserstein space,

$$
{\color{blue}\frac{\partial \rho_t}{\partial t} - \Delta\rho = 0}.\tag{HE}
$$


**$f$-Divergence** When $\mathcal{F}$ is defined to be the $f$-divergence between $\rho_t$ and a fixed target density $\mu$, i.e.,

$$
\mathcal{F} \triangleq \int f\left(\frac{\rho(x)}{\mu(x)}\right)\mu(x)dx,
$$

where $f$ is a twice-differentiable convex function with $f(1) = 0$, the gradient flow is given by the following PDE,

$$
\frac{\partial \rho_t}{\partial t} - \nabla\cdot\left(\rho(x)\nabla f'\left(\frac{\rho(x)}{\mu(x)}\right)\right) = 0.
$$

For the special case of the KL divergence, i.e., $f = r\log r$, we recover the Fokker-Plank Equation,

$$
{\color{blue}\frac{\partial \rho_t}{\partial t} + \nabla\cdot(\rho(x)\nabla\log\mu(x)) - \Delta\rho(x) = 0}.\tag{FPE}
$$

## Concluding Remarks

Gradient flows have been a popular tool in the analysis of PDEs. Recently, gradient flows of various distances/divergences used in machine learning have been proposed [3, 4, 5, 6, 7] and have been used for generative modeling [4, 5, 7] and sample refinement in generative models [8], among other cool applications [5]. This article is my attempt at providing some background for readers unfamiliar with gradient flows.

## References

<div style="font-size: 80%" markdown="1">

[1] Filippo Santambrogio. {Euclidean, metric, and Wasserstein} gradient flows: an overview. Bulletin of Mathematical Sciences, 7(1), 2017.     

[2] Brittany Hamfeldt. Optimal Transport - Gradient Flows in the Wasserstein Metric. [Video Lecture](https://www.youtube.com/watch?v=zzGBxAqJV0Q), 2019.   

[3] Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum mean discrepancy gradient flow. In NeurIPS, 2019.    

[4] Antoine Liutkus, Umut Simsekli, Szymon Majewski, Alain Durmus, and Fabian-Robert Stoter. Sliced-Wasserstein flows: Nonparametric generative modeling via optimal transport and diffusions. In ICML, 2019.    

[5] Youssef Mroueh, Tom Sercu, and Anant Raj. Sobolev descent. In AISTATS, 2019.    

[6] Qiang Liu. Stein variational gradient descent as gradient flow. In NeurIPS, 2017.      

[7] Yuan Gao, Yuling Jiao, Yang Wang, Yao Wang, Can Yang, and Shunkang Zhang. Deep generative learning via variational gradient flow. In ICML, 2019.    

[8] Abdul Fatir Ansari, Ming Liang Ang, and Harold Soh. Refining Deep Generative Models via Wasserstein Gradient Flows. arXiv preprint arXiv:2012.00780, 2020.
</div>

------------
### Footnotes
<span style="font-size: 70%; color: gray;"><sup>1</sup> Note that we no longer require $F$ to be convex but other regularity conditions (lower semi-continuity and boundedness from below) are required for the existence of minimizers.</span>

<span style="font-size: 70%; color: gray;"><sup>2</sup> $T_{\sharp}\mu = \nu$ means that if we apply the map $T$ to samples from $\mu$ we get samples that are distributed according to $\nu$.</span>

<span style="font-size: 70%; color: gray;"><sup>3</sup> Please check [this article](http://abdulfatir.com/Wasserstein-Distance/) for an intuitive introduction to the $c$-transform of a function. </span>

<span style="font-size: 70%; color: gray;"><sup>4</sup> The first variation is defined up to additive constants since $\chi$ is a zero-mean measure (it's a difference of two probability measures).</span>
