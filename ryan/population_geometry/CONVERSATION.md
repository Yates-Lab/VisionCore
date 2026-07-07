Ryan: 

I am considering the example of the derivatives of a rigid image under translation I(x,y,t) = I_0(x+D_x(t),y+D_y(t)). This is a relevant formulation for the effects of small eye movements (like drift) on a static image. I am thinking about the problem of image registration under noisy observations (like in the nervous system). First, I'd like to know the analytical form of the derivative of I with respect to t and also what that can teach us about how the statistics of D interact with I_0 to produce transients in the image. Then, we should think about what it means to do inference about I_0 given just noisy observation of I.

---

Opus:

## The temporal derivative

By the chain rule, with $\mathbf{D}(t) = (D_x, D_y)$ and the shifted coordinate $\boldsymbol\xi = (x+D_x(t),\, y+D_y(t))$:

$$\partial_t I(x,y,t) = \nabla I_0(\boldsymbol\xi)\cdot \dot{\mathbf D}(t) = I_{0,x}(\boldsymbol\xi)\,\dot D_x + I_{0,y}(\boldsymbol\xi)\,\dot D_y.$$

This is just the brightness-constancy / optic-flow constraint: the transient at any retinal location is the projection of the eye *velocity* onto the local image *gradient*. Three immediate consequences:

- **No gradient, no transient.** Flat regions produce zero temporal signal no matter how the eye moves. Only the gradient-aligned component of velocity is visible — the aperture problem in miniature. Edges and high-SF content do all the work.
- **The two factors multiply.** Conditioning on position so $\nabla I_0$ is fixed, $\langle(\partial_t I)^2\rangle = \dot{\mathbf D}^\top\langle\dot{\mathbf D}\dot{\mathbf D}^\top\rangle^{-1}\!\cdots$ — concretely, for isotropic velocity with variance $\sigma_v^2$ per axis, $\langle(\partial_t I)^2\rangle = \sigma_v^2\,|\nabla I_0|^2$. Transient power scales with squared gradient magnitude and eye speed.
- **Spatial structure is recoded as timing.** The instant a high-gradient feature sweeps across a receptor sets the timing of its transient. This is the "temporal encoding of spatial information" idea — drift turns a spatial map into a temporal code.

## How the statistics of D shape the transients

The cleanest way to see the D–$I_0$ interaction is in the Fourier domain. Writing $I(\mathbf r,t)=\int \hat I_0(\mathbf k)\,e^{i\mathbf k\cdot\mathbf r}\,e^{i\mathbf k\cdot\mathbf D(t)}\,d\mathbf k$, each spatial component is phase-modulated in time by $e^{i\mathbf k\cdot\mathbf D(t)}$. If drift is 2-D Brownian with diffusion constant $\kappa$ (the standard first model), increments are Gaussian and the temporal autocorrelation of that modulation is the Gaussian characteristic function:

$$\big\langle e^{i\mathbf k\cdot[\mathbf D(t)-\mathbf D(t')]}\big\rangle = e^{-\kappa|\mathbf k|^2\,|t-t'|}.$$

So each spatial frequency decorrelates exponentially at rate $\kappa|\mathbf k|^2$ — **high SFs flicker fast, low SFs slowly**. The resulting spatiotemporal power spectrum is

$$P(\mathbf k,\omega) = |\hat I_0(\mathbf k)|^2\cdot\frac{2\kappa|\mathbf k|^2}{\omega^2 + (\kappa|\mathbf k|^2)^2}.$$

Here's the payoff. Natural images have $|\hat I_0(\mathbf k)|^2 \sim A/|\mathbf k|^2$. Above the knee $\omega \gg \kappa|\mathbf k|^2$, the temporal factor $\to 2\kappa|\mathbf k|^2/\omega^2$, and the $|\mathbf k|^2$ **cancels the $1/|\mathbf k|^2$ of the image**:

$$P(\mathbf k,\omega)\;\approx\;\frac{2A\kappa}{\omega^2}.$$

Drift acts as a *whitening filter*: it flattens the spatial spectrum across the mid/high-SF band, equalizing power and decorrelating the input — efficient-coding-optimal in the Atick–Redlich/Barlow sense. This is essentially the Kuang–Poletti–Victor–Rucci (2012) result, and the geometric mechanism is exactly the $i\mathbf k$ (gradient) factor in your chain rule getting squared into power.

**One subtlety worth flagging given the "noisy/stochastic D" framing.** If $\mathbf D$ is genuinely Brownian, $\dot{\mathbf D}$ doesn't exist as a function — the expression $\partial_t I = \nabla I_0\cdot\dot{\mathbf D}$ is formal, shorthand for the increment statistics. The honest object is the Itô differential:

$$dI = \underbrace{\nabla I_0(\boldsymbol\xi)\cdot d\mathbf D}_{\text{martingale (the transients)}} \;+\; \underbrace{\kappa\,\nabla^2 I_0(\boldsymbol\xi)\,dt}_{\text{Itô drift}}.$$

The second-order term is a Laplacian. Taking expectations kills the martingale part and leaves $\partial_t\langle I\rangle = \kappa\nabla^2\langle I\rangle$ — the **heat equation**. Averaging over drift trajectories *blurs* $I_0$ with a Gaussian of width $\sqrt{2\kappa t}$. So the smooth (ensemble-mean) part of the signal is being destroyed by the very motion that helps you; all the high-SF information lives in the fluctuating martingale term. That tension is what makes the inference problem interesting.

## Inference about $I_0$ from noisy observations of $I$

Observe $y(\mathbf r,t) = I_0(\mathbf r + \mathbf D(t)) + \eta$, with $\mathbf D$ unknown (or only weakly constrained by efference copy). You want $p(I_0\mid y)=\int p(I_0,\mathbf D\mid y)\,d\mathbf D$. Some structural points:

**It's a blind-deconvolution / SLAM problem.** You must jointly estimate the *map* ($I_0$) and the *trajectory* ($\mathbf D$), and they're coupled: you register frames using $\nabla I_0$, but you don't know $I_0$ until you've registered. Hence the natural algorithm is coordinate ascent / EM — estimate $\mathbf D$, de-translate and average, re-estimate. Burak, Rokni, Meister & Sompolinsky (2010, PNAS) showed a neural decoder can do exactly this jointly from noisy retinal spikes, so drift does not preclude high acuity — a Bayesian observer turns it into an asset.

**The information geometry follows directly from Part 1.** Since $\partial_{\mathbf D}\, y = \nabla I_0$, the Fisher information about displacement is $\mathcal I_{\mathbf D}\propto |\nabla I_0|^2/\sigma_\eta^2$ — **you localize the eye best precisely where the image has strong edges**, the same places that generate the transients. Conversely, information about $I_0$ at frequency $\mathbf k$ accumulates as the trajectory samples different phases of that component. The motion buys you *multiple sub-receptor-spaced looks*, which is the basis of super-resolution: drift lets you recover SFs above the photoreceptor-lattice Nyquist limit. This is why motion helps rather than hurts.

**Two failure modes / identifiability limits, both predicted by the spectrum above:**

1. *Gauge freedom.* You can shift $I_0$ and absorb it into a constant offset of $\mathbf D$ — absolute position is unobservable. Only relative structure is recoverable.
2. *Low SFs are starved.* The temporal modulation power $\propto |\mathbf k|^2$ vanishes as $\mathbf k\to 0$, so Brownian drift barely modulates low spatial frequencies — they decorrelate too slowly to separate from a slowly varying signal, and pure trajectory-integration accumulates random-walk error there. Low-SF content must come from the prior, from microsaccades/larger excursions, or from efference copy. Drift is a mid/high-SF instrument.

The clean conceptual summary: marginalizing $\mathbf D$ naively reintroduces exactly the heat-equation blur from the Itô term, so a good observer instead *estimates* $\mathbf D$ (using edge-localized Fisher information) and conditions on it, recovering the high-SF martingale signal that the whitening makes uniformly available. Efference copy enters as an informative prior on $p(\mathbf D)$ that tightens the joint estimate but, per Burak et al., isn't strictly required.

If it's useful I can push any branch further — e.g., write out the EM updates for the joint estimator, work the Cramér–Rao bound on acuity as a function of $\kappa$ and $\sigma_\eta$, or do the same spectral analysis for a correlated (Ornstein–Uhlenbeck or self-avoiding) drift model instead of pure Brownian, which changes the whitening band.

--- 

Ryan: 

There are a few results in the vein of Burak, Rokni, Meister & Sompolinsky that show that jointly estimating I_0 and D improve reconstruction, and this it the basis for super-resolution in phone cameras. However, not much work has been done on how primary visual cortex may implement part of the super-resolution algorithm. I have lots of recordings from foveal primary visual cortex freely viewing  natural images. We have highly precise eye tracking, so assume we have prefect information about where the eye is pointed and what the visual input was alongside populations of neurons' responses. The question then is how can we take the existing work in this field and learn something new about V1. We have access to laminar information about where the cells were as well. What might be a geometric interpretation of the need to estimate D and I_0 in a population of cells. What might be traces of this function running in neural tissue. Make sure to self-critique any hypotheses you return to me and make sure they stand up to scrutiny.

---

Opus:

The most useful thing I can give you is a single geometric object that the whole problem collapses onto, and then a disciplined list of what would and wouldn't count as evidence that V1 is touching it.

## The geometry: a fiber bundle whose base is content and whose fiber is gaze

Translation is a group acting on images. Hold a foveal patch $I_0$ fixed and let $\mathbf D$ range over its drift excursion; the retinal inputs $I_0(\mathbf r + \mathbf D)$ trace a 2-D **orbit** — the translation orbit of that patch. Your V1 population maps this orbit to a 2-D manifold in firing-rate space $\mathbb R^N$. Now decompose the variation:

- **Along the orbit** (change $\mathbf D$, fix $I_0$): the nuisance you want to estimate and discount.
- **Across orbits** (change $I_0$): the content you want to recover.

The tangent plane to the orbit at any point is spanned by $\partial_{D_x}\mathbf r$ and $\partial_{D_y}\mathbf r$ — which are exactly the population responses to the **gradient images** $I_{0,x}, I_{0,y}$ pushed through the neural transfer function. That is the chain-rule fact from before, now living in neural space: $\partial_{\mathbf D}\mathbf r = J$, and $J$'s columns *are* the encoded gradient. So the abstract claim "$\partial_{\mathbf D} y = \nabla I_0$" becomes a concrete geometric assertion: **the directions in population space along which activity flows under drift are the directions that encode the image gradient.**

Stack this over all patches and you get a fiber bundle. The base is content (image modulo translation, the quotient $\text{Image}/\mathbb R^2$); the fiber over each base point is gaze position $\mathbf D$. Estimating $\mathbf D$ = localizing yourself *along the fiber*, which requires the generators (the tangent/gradient subspace). Estimating $I_0$ = identifying *which base point* you're on, i.e., projecting out the group action. Super-resolution is the statement that motion *along the fiber* lets you refine the *base* estimate beyond the photoreceptor lattice — you accumulate sub-pixel looks because the fiber coordinate sweeps continuously while the base stays fixed. The thing that makes this work — the rule for transporting a content estimate from one gaze position to another so they can be averaged — is the **connection** on the bundle, and *that* is the registration computation. If V1 implements any of the algorithm, the connection is what's instantiated in tissue.

This gives you a precise, falsifiable linear test rather than a metaphor: locally, $J = \partial \mathbf r/\partial \mathbf D$ should be approximately **rank 2**, and its two singular vectors should match the population encodings of $I_{0,x}$ and $I_{0,y}$ measured independently. Rank and alignment are both checkable with your data and your eye traces.

## Candidate traces in tissue — each with the reason it might be nothing

**1. The Jacobian $\partial \mathbf r/\partial \mathbf D$ is low-rank and gradient-aligned, and $\mathbf D$ is linearly decodable at near-Fisher precision.** You'd regress population rates against your measured $\mathbf D(t)$, check rank $\approx 2$, and check the singular vectors against the independently estimated gradient encoding. The Fisher information for $\mathbf D$ should track $|\nabla I_0|^2/\sigma^2$ — you localize gaze best over high-contrast structure.

*Critique — and this one is fatal if you stop here:* decodability of $\mathbf D$ is **necessary but vacuous** as evidence of a registration *function*. Any faithful retinotopic filter lets you decode the position of a known image; that's just "V1 represents the image and the image moved." The trivial feedforward null predicts every part of H1 — low rank, gradient alignment, Fisher-matched precision — with no estimation happening anywhere. So H1 is the right *sanity check* (if it fails, the geometry above is wrong) but it cannot by itself support any claim about V1 computing $\mathbf D$. The science is entirely in separating passive representation from active estimation, which is H4.

**2. Simple/complex division of labor implements the fiber/base split.** Tempting: phase-sensitive simple cells carry the tangent (gradient → registration signal), phase-invariant complex cells carry translation-tolerant content.

*Critique:* I think this is seductive and mostly wrong as stated, and worth stating why so you don't chase it. Complex-cell invariance is local to the RF, so it gives you no global handle on the drift excursion; the energy model makes complex cells a *nonlinear function of the same simple inputs*, not an independent channel, so they don't carry orthogonal information; and quantitatively, drift amplitude (a few arcmin) is small relative to foveal RF size, so the "feature sweeps across the field" picture is marginal. Worse, the direction is backwards for super-resolution: phase invariance *discards* the very registration signal you need. If there's anything here it's graded, not a clean dichotomy, and I'd demote it.

**3. A laminar gradient: input layers carry the motion-corrupted feedforward image, deeper layers carry a more stabilized estimate.** Predict $\mathbf D$-decodability and instantaneous-input fidelity highest in 4C, content stability (I₀-like, $\mathbf D$-invariant) rising in 2/3 and infragranular layers with a lag.

*Critique:* three confounds, each of which can fake the whole pattern. (i) Complex cells are denser outside 4C, so "more invariant deeper" may just be the simple→complex transform from H2, not accumulation. (ii) Cross-laminar lags are an automatic consequence of the feedforward hierarchy and synaptic delay — a lag is not evidence of integration *toward an estimate*. (iii) The sign isn't even clearly predicted: an acuity-preserving computation might want to *keep* high-SF phase deep, not throw it away, so "more invariant deeper" could be the opposite of what super-resolution wants. Use the laminar axis as a place to localize an effect you've *already* established by other means — don't let it be the existence proof.

**4. (The one I'd actually bet on) Motion-compensated accumulation: drift-aligned, drift-scaled variance suppression within a fixation.** This is the test that distinguishes active estimation from passive filtering. If the network is averaging looks across gaze positions to stabilize the content estimate, it must *cancel* the $\mathbf D$-induced excursion — so within a fixation the population covariance should shrink **specifically along the orbit-tangent (gradient) direction**, and the amount of shrinkage should **scale with the drift amplitude actually experienced** on that fixation. A pure feedforward filter passes $\mathbf D$-variation straight through; an accumulator settles onto the orbit.

*Critique:* the obvious confound is adaptation — V1 variance shrinks within a fixation from contrast/spike-rate adaptation regardless of any registration. The escape is geometric and is what makes this falsifiable rather than hand-wavy: adaptation is roughly *isotropic* in stimulus space and *blind to $\mathbf D$*, whereas motion compensation predicts shrinkage **aligned with the measured tangent direction** and **graded by per-fixation drift magnitude**. Those two conditionings are not something adaptation produces. The residual risk is that the effect is small and you're variance-limited; this needs many repeats over matched patches, which free viewing makes awkward (you rarely revisit the same patch at the same content). You may need to construct matched-patch ensembles post hoc using the eye trace, which you're uniquely positioned to do.

**5. Stabilized-frame / predictive signature.** If part of the connection is implemented, some V1 responses should be referenced to a world/object frame rather than a purely retinal one at the *drift* scale — a micro-analog of peri-saccadic remapping. Concretely: current response carries information about the drift-*predicted* input, beyond what the instantaneous retinal input explains.

*Critique:* remapping at the drift scale (arcmin, smooth) is a much weaker, harder-to-detect thing than the classic saccadic version, and "predicts future input" can be mimicked by any temporal low-pass / momentum in the response. To make it real you need the predictive component to follow the *specific* $\mathbf D$ trajectory, not just generic temporal smoothing — i.e., conditioning on the actual drift direction should change the predicted-input term, which smoothing can't do. Given your L6 corticothalamic interests, this is also where a substrate would plausibly live (the L6→LGN/TRN loop as the gating that compensates), which is a nice mechanistic hook but I'd treat that as a downstream hypothesis, not a starting assumption.

**6. Drift-matched spatiotemporal coding.** From the whitening result, an efficient registration front-end should match its SF×TF coupling to the drift spectrum: higher-SF neurons should show faster temporal dynamics tracking the $\kappa|\mathbf k|^2$ knee.

*Critique:* SF–TF inseparability in V1 is already known and usually chalked up to generic motion/direction selectivity. To make this a *drift* claim and not a re-description of direction selectivity, the coupling has to match *that animal's measured* drift diffusion constant — ideally tracking it across eccentricity (where drift's relative contribution changes) or across behavioral state. Without the link to the individual's oculomotor statistics it's not a new result.

## What's genuinely new here, stated honestly

The representation of the *ingredients* — encoded gradient (tangent) plus content (base) — is almost certainly present in V1 trivially, because a faithful retinotopic code contains them by construction. That is not a discovery. The open, contestable, publishable question is whether V1 performs any of the *connection* — the motion-compensated transport that turns multiple gaze positions into one refined estimate — or whether it is a clean retinotopic filter and all joint estimation is downstream. H1, H2, H3, H6 are, at best, consistency checks or descriptions of the substrate; H4 (drift-aligned, drift-scaled, tangent-specific variance suppression) and the strict version of H5 (trajectory-specific prediction) are the only signatures I can construct that a passive feedforward model cannot fake. I'd put the weight of the experiment there, use the fiber-bundle geometry to define the directions you measure against, and use the laminar axis only to localize an effect once it's established.

The clean one-line framing for the grant or talk: *drift defines a fiber over each foveal image patch; V1 unavoidably encodes the fiber tangent (the gradient) and the base (the content), so the empirical question is not whether V1 represents $\mathbf D$ and $I_0$ but whether it implements the connection that binds looks across the fiber — and the only model-distinguishing fingerprint of that connection is tangent-aligned, drift-amplitude-graded stabilization that adaptation cannot mimic.*

I can take any of these further: write the explicit per-fixation estimator for the tangent-aligned variance test (including the matched-patch construction from your eye traces and the adaptation-isotropy null), derive the Cramér–Rao bound on the base estimate as a function of $\kappa$, $\sigma_\eta$, and number of looks to tell you how big an effect to expect, or work the generator/connection formalism into something you could fit as a recurrent dynamics model against the population.
