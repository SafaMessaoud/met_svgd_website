(sec5)=
# Results

## Energy-Based Models

Energy-based models (EBMs) are generative models that learn a density $p_\phi$ parametrized by an energy function $f_\phi$, so that $p_\phi(x) = \bar p_\phi(x) / Z_\phi$, with $\bar p_\phi(x)=\exp(f_\phi(x))$.
Typically, EBMs are trained via maximum likelihood estimation (MLE) by estimating the gradient of the maximum likelihood loss, since direct computation of the loss is not feasible due to the intractability of $Z_\phi$.
However, when a sampler with a tractable density $q_\theta$ is given, a tight lower bound on the MLE loss can be computed:
:::{math}
:enumerated: false

\mathcal{L}_\text{ELBO}(\phi, \theta)
=
\mathbb{E}_{x \sim q_\theta}\left[ \log \bar p_\phi(x) \right]
-
\mathbb{E}_{x \sim p_\text{data}}\left[ \log \bar p_\phi(x) \right]
+
\mathcal{H}(q_\theta).
:::

The EBM is then trained by alternating between maximizing $\mathcal{L}_\text{ELBO}(\phi, \theta)$ with respect to $\theta$ to further tighten the lower bound, and minimizing it with respect to $\phi$.

In [](#cifar), we show the Fréchet Inception Distance (FID) averaged over 5 random seeds for CIFAR10 image generation.
We compare against P-SVGD [@s2ac] and Glow (Normalizing Flow) [@glow].

:::{figure} figures/fid.svg
:label: cifar
:class: flex flex-col items-center justify-center
FID on CIFAR-10; bold marks changes between configurations.
:::

Removing either the [trace-of-Hessian term](#sec3-tr) or the step-size bound causes the training to diverge, as the violet and gray curves show.
[](#cifar-score) shows that adding the trace-of-Hessian term produces smoother energy landscapes that are much easier to sample from.
The step-size bound plays a different but equally important role, as it ensures that the entropy estimation remains valid.

:::{figure} figures/cifar_score.svg
:label: cifar-score
:class: flex flex-col items-center justify-center
Smoothness of the learnt EBM as measured by $||\nabla \log \bar p_\phi||$ throughout training iterations.
The configurations with the best FID ([](#cifar)) exhibit smoother landscapes.
:::

Replacing the median heuristic bandwidth, $\sigma_\text{med}$, with a learnable bandwidth greatly improves stability and leads to significantly better FID scores compared to P-SVGD (green versus orange).
As shown in [](#cifar-sigma), the median heuristic bandwidth is on average more than an order of magnitude larger than the learned value $\sigma_{\theta_2}$.
This overly large bandwidth causes particles to become spuriously correlated, hurting sample quality.

Learning the step size as well (red curve) allows the method to converge to the target distribution much faster.
[](#cifar-step) illustrates that the learned step size $\epsilon_{\theta_3}$ is much larger than the fixed step size $\epsilon$.

:::{figure} figures/cifar_sigma_step_legend.svg
:class: mb-0
:label: sigma-step-legend
:enumerated: false
:::

::::{figure}
:label: cifar-sigma-step
:class: flex flex-row flex-wrap items-center justify-center mt-0

:::{figure} figures/cifar_sigma.svg
:class: flex flex-col items-center justify-center
:label: cifar-sigma
Kernel bandwidth across training iterations.
:::

:::{figure} figures/cifar_step.svg
:class: flex flex-col items-center justify-center
:label: cifar-step
Step-size across training iterations
:::

Learnt kernel bandwidth and step-size across training iterations.
The trends exhibited by the learnt kernel bandwidth and step-size in MET-SVGD configurations are significantly different from $\sigma_\text{med}$ and constant $\epsilon$.
::::

Using an adaptive number of update steps $L_c$ further stabilizes training, as shown by the brown curve.

In contrast to the other MET-SVGD optimizations, experiments that incorporate MH diverge, as indicated by the pink curve.
This instability is not inherent to MET-SVGD itself, but instead comes from the interaction between MH rejection dynamics and a the loss function, $\mathcal{L}_\text{ELBO}$, not being lower-bounded.
The stability of the training objective depends critically on the quality of the samples used to estimate the expectation with respect to the sampler.
When the energy landscape is highly complex, MH acceptance rates can become very low [@cifar-rej], preventing particles from moving toward high-density regions of the target distribution.
As a result, sample quality degrades and training eventually diverges.

:::{figure} figures/cifar_mh_rej.svg
:label: cifar-rej
:class: flex flex-col items-center justify-center
MH rejection rate across training iterations.
:::

Glow-NF exhibits a similar failure mode, struggling to produce good samples early in training and ultimately diverging as well.

## Maximum Entropy Reinforcement Learning

MaxEnt RL learns a stochastic policy over actions given a state by maximizing the sum of expected rewards and entropies:
:::{math}
:enumerated: false

\pi^*_\theta
=
\underset{\pi_\theta}{\mathrm{argmax}}
\sum_t
\mathbb{E}_{(s_t, a_t)} \left[
r(s_t, a_t)
+
\alpha \mathcal{H}(\pi_\theta(\cdot | s_t))
\right]
:::

We model the sampler and estimate the entropy using the MET-SVGD framework and compare against SAC [@sac], which models the policy as a diagonal Gaussian, SAC-NF, which models the policy as an auto-regressive normalizing flow [@maf], and P-SVGD [@s2ac].
In @walker, we report the average return on $10$ rollouts every $1000$ steps on Walker2d-v2 averaged over $5$ seeds.

:::{figure} figures/walker_return_legend.svg
:class: mb-0
:label: walker-legend
:enumerated: false
:::

::::{figure}
:label: walker
:class: flex flex-row flex-wrap items-top justify-center mt-0

:::{figure} figures/walker_return_curve.svg
:class: flex flex-col items-center justify-center
:label: walker-curve
Return IQM across training iterations.
:::

:::{figure} figures/walker_return_bar.svg
:class: flex flex-col items-center justify-center
:label: walker-bar
Return IQM at the end of training.
:::

Return IQM on Walker2d-v2.
::::

Learning the kernel bandwidth (orange) and adding the trace-of-Hessian correction (green) both lead to substantial improvements over the P-SVGD baseline.
In contrast, removing the bound on the step size (pale green) causes the method to diverge, which is expected.

Learning the step size without the trace-of-Hessian term (peach) performs worse than all baselines.
In this case, particles tend to drift into non-smooth regions of the landscape [@walker-score].

:::{figure} figures/walker_score.svg
:label: walker-score
:class: flex flex-col items-center justify-center
Smoothness as measured by $||\nabla \log \bar p_\phi||$ throughout training iterations.
:::

The best results come from incorporating MH with an $\epsilon$-greedy schedule (purple):
applying MH with high probability later in training and low probability early on helps balance exploration in the beginning with exploitation as learning progresses.
These improvements become even more pronounced when we increase the number of particles from 10 to 64 (brown).