# 1 Intro
* ODE is modeled as the limit of the state transition equation defined below, by taking smaller steps and increasing the number of layers.

$$
h_{t + 1} = h_{t} + f(h_{t}, \theta_{t})
$$
* Parametrizing as ODE we get the neural network specify the derivative of the hidden state. $h(0)$ serves as the input and $h(T)$ the solution to this ODE. 

$$
\begin{aligned}
\frac{d\mathbf{h}(t)}{dt} &= f(\mathbf{h}(t), t, \theta)
\end{aligned}
$$
* Benefits of defining and evaluating the models as ODE
	* Memory efficiency
	* Adaptive computation
	* Scalable and invertible normalizing flows
	* Continuous time series models

# 2 Reverse-mode automatic differentiation of ODE solutions

## 2.1 Proof to Adjoint Method 
### 2.1.1 Continuous Backpropagation

Let us define the loss as

$$
\begin{align} \\
L(\mathbf{z}(t_{1})) &= L\left(\mathbf{z}(t_{0}) + \int_{t_{0}}^{t_{1}} f(\mathbf{z}(t), t, \theta)dt\right) = L\left(ODESolve(\mathbf{z}(t_{0}), f, t_{0}, t_{1}, \theta)) \right) \tag{1}

\end{align}
$$

Let $\mathbf{z(t)}$ follow the differential equation $\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)$ where $\theta$ are the parameters. If we define an adjoint state which is the gradient wrt hidden state at a specified time.

$$
\mathbf{a}(t) = \frac{dL}{d\mathbf{z}(t)} \tag{2}
$$
Then this adjoint state follows the differential equation

$$
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)\frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} \tag{3}
$$

Since the neural network models the derivative of the hidden state, we have

$$
\mathbf{z}(t + \epsilon) = \mathbf{z}(t) + \int_{t}^{t + \epsilon} f(\mathbf{z}(\tau), \tau, \theta) d\tau = T_{\epsilon}(\mathbf{z}(t), t) \tag{4}
$$

Assuming that the function $f(\mathbf{z}(t), t, \theta)$ is a smooth function then we have the Taylor series expansion as 

$$
f(\mathbf{z}(\tau), \tau, \theta) = f(\mathbf{z}(t), t, \theta) + f'(\mathbf{z}(t), t, \theta)(\tau - t) + \frac{1}{2}f''(\mathbf{z}(t), t, \theta)(\tau - t)^2 + H.O.T \tag{5}
$$

Integrating this leads us to have the Taylor series expansion for integral

$$
\begin{align}
T_{\epsilon}(\mathbf{z}(t), t) &= \mathbf{z}(t) + \int_{t}^{t + \epsilon}f(\mathbf{z}(\tau),\tau,\theta)d\tau \\

T_{\epsilon}(\mathbf{z}(t), t) &= \mathbf{z}(t) + f(\mathbf{z}(t), t, \theta)\cdot \epsilon + f'(\mathbf{z}(t), t, \theta)\cdot \frac{\epsilon^2}{2} + f''(\mathbf{z}(t), t, \theta)\cdot\frac{\epsilon^3}{6} + H.O.T \tag{6}
\end{align}
$$

In case of standard/traditional neural networks, the gradient of the hidden layer $\mathbf{h}_{t}$ depends on the gradient from the next layer $\mathbf{h}_{t + 1}$. Using chain rule we have

$$
\frac{dL}{d\mathbf{h}_{t}} = \frac{dL}{d\mathbf{h}_{t+1}}\cdot \frac{d\mathbf{h}_{t+1}}{d\mathbf{h}_{t}} \tag{7}
$$

In case of continuous hidden state, applying the chain rule leads to 

$$
\begin{align}
\frac{dL}{d\mathbf{z}(t)} &= \frac{dL}{d\mathbf{z}(t+\epsilon)}\cdot \frac{d\mathbf{z}(t + \epsilon)}{d\mathbf{z}(t)} \\
& \text{or} \\
\mathbf{a}(t) &= \mathbf{a}(t+\epsilon)\cdot \frac{\partial T_{\epsilon}(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)} \tag{8}
\end{align}
$$

The derivative of equation $(7)$ using the definition of derivative is

$$
\begin{align}
\frac{d\mathbf{a}(t)}{dt} &= \lim_{ \epsilon \to 0^+ } \frac{\mathbf{a}(t + \epsilon) - \mathbf{a}(t)}{\epsilon}  \\
&= \lim_{ \epsilon \to 0^+ }  \frac{\mathbf{a}(t + \epsilon) - \mathbf{a}(t+\epsilon)\cdot \frac{\partial T_{\epsilon}(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)}}{\epsilon} \\
&= \lim_{ \epsilon \to 0^+ } \frac{\mathbf{a}(t + \epsilon) - \mathbf{a}(t+\epsilon)\cdot \frac{\partial}{\partial \mathbf{z}(t)} (\mathbf{z}(t) + \epsilon f(\mathbf{z}(t), t, \theta) + \mathcal{O}(\epsilon^2))}{\epsilon} \\
&= \lim_{ \epsilon \to 0^+ } \frac{\mathbf{a}(t + \epsilon) - \mathbf{a}(t+\epsilon)\cdot \left( I + \epsilon \cdot \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} + \mathcal{O}(\epsilon^2)\right)}{\epsilon} \\
&= \lim_{ \epsilon \to 0^+ } {- \mathbf{a}(t+\epsilon)\cdot \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} + \mathcal{O}(\epsilon)} \\
\frac{d\mathbf{a}(t)}{dt} &= -\mathbf{a}(t) \cdot \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} \tag{9}
\end{align}
$$

Equation $(9)$ is similar to backpropagation defined in Equation $(7)$. We start with the last time step $t_N$ and move backwards in time solving the ODE. $\mathbf{a}(t_{N})$ becomes our initial condition and $\mathbf{a}(t_{0})$ our solution to the reverse ODE.

$$
\begin{align}
\mathbf{a}(t_{N}) &= \frac{dL}{d\mathbf{z}(t_{N})} \tag{10} \\
 \\
\mathbf{a}(t_{0}) &= \mathbf{a}(t_{N}) + \int_{t_{N}}^{t_{0}} \frac{d \mathbf{a}(t)}{dt} dt \\
\\
\mathbf{a}(t_{0}) &= \mathbf{a}(t_{N}) - \int_{t_{N}}^{t_{0}} \mathbf{a}(t)^T \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}dt \tag{11}
\end{align}
$$


### 2.1.2 Gradient wrt $\theta$ and $t$

Note that $\frac {\partial \theta(s)}{\partial s} = 0$ and  $\frac{dt(s)}{ds} = 1$. Here we are using $s$ to denote the time and $t$ as another variable that follows $s$.

Rewriting the derivative as

$$
\frac{d}{ds}
\begin{bmatrix}
\mathbf{z} \\
\theta \\
t
\end{bmatrix}(s) = f_{aug}([\mathbf{z, \theta, t}]) := \begin{bmatrix}
f(\mathbf{z}, \theta, t) \\
0 \\
1
\end{bmatrix} \tag{12}
$$

We will then have the augmented adjoint as

$$
\mathbf{a}_{aug} = \begin{bmatrix}
\mathbf{a} \\
\mathbf{a}_{\theta} \\
\mathbf{a}_{t}
\end{bmatrix}, \quad \mathbf{a}_{\theta}(s) := \frac{dL}{d\theta(s)}, \mathbf{a}_{t}(s) = \frac{dL}{dt(s)} \tag{13}
$$

$$
\frac{\partial f_{aug}}{\partial[\mathbf{z}, \theta, t]} = \begin{bmatrix}
\frac{\partial f}{\partial \mathbf{z}} & \frac{\partial f}{\partial\theta} & \frac{\partial f}{\partial t} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0}
\end{bmatrix} (t) \tag{14}
$$

Using equation $(14)$ in equation $(9)$ we get

$$
\begin{align}
\frac{d\mathbf{a}_{aug}(s)}{ds} &= -\begin{bmatrix}
\mathbf{a}(s) & \mathbf{a}_{\theta}(s) & \mathbf{a}_{t}(s)
\end{bmatrix} \frac{\partial f_{aug}}{\partial \begin{bmatrix}
\mathbf{z}; \theta; t
\end{bmatrix}}(s) \\
&= -\begin{bmatrix}
\mathbf{a} \frac{\partial f}{\partial \mathbf{z}} & \mathbf{a} \frac{\partial f}{\partial \theta} &  \mathbf{a}\frac{\partial f}{\partial t}
\end{bmatrix} (s) \tag{15}
\end{align}
$$

By introducing the new state variable $t$, $f_{aug}$ now depends on $t$ and not on $s$ making it time-invariant. While the previous formulation required us to compute the gradient wrt $\theta$ separately, but with this new formulation we can get the derivative in one step. We combine $\mathbf{z}, \theta$ and $t$ into a state and pass it to the ODE solver to compute the gradients.

In order to compute the gradients wrt $\theta$ we use the below equation $(16)$ by setting $\mathbf{a}_{\theta}(s_{N}) = 0$.

$$
\frac{dL}{d\theta} = \mathbf{a}_{\theta}(s_{0}) = - \int_{s_{N}}^{s_{0}}\mathbf{a}(s) \frac{\partial f(\mathbf{z}(s), s, \theta)}{\partial\theta} ds \tag{16}
$$

The gradient of $L$ wrt $t_{N}$ can be written using chain rule 

$$
\begin{align}
\frac{dL}{dt_{N}} &= \frac{dL}{d\mathbf{z}(t_{N})} \cdot \frac{d\mathbf{z}(t_{N})}{dt_{N}} \\
&= \mathbf{a}(t_{N}) \cdot \frac{d\mathbf{z}(t_{N})}{dt_{N}} \\
&= \mathbf{a}(t_{N}) \cdot \frac{d\mathbf{z}(t_{N})}{dt_{N}} \cdot 1 \\
&= \mathbf{a}(t_{N}) \cdot \frac{d\mathbf{z}(t_{N})}{dt_{N}} \cdot \frac{dt_{N}}{ds_{N}} \\
&= \mathbf{a}(t_{N}) \cdot \frac{d\mathbf{z}(t_{N})}{ds_{N}} \\
\frac{dL}{dt_{N}} &= \mathbf{a}(t_{N}) f(\mathbf{z}(s_{N}), s_{N}, \theta) \tag{17}
\end{align}
$$

Similarly we have gradient of L wrt $t_{0}$ as

$$
\frac{dL}{dt_{0}} = \mathbf{a}_{t}(t_{0}) = \mathbf{a}_{t}(t_{N}) - \int_{t_{N}}^{t_{0}} \mathbf{a}(s) \frac{\partial f(\mathbf{z}(s), s, \theta)}{\partial s}ds \tag{18}
$$

**Note**:
1. We know that the loss $L$ is directly dependent on $\mathbf{z}(s_{N})$ and not $\theta$. Hence loss does not change by perturbing $\theta$ at $s_{N}$. But $\theta$ at $s_{N} - \epsilon$ influences the value of the loss through $\mathbf{z}(s_{N} - \epsilon)$ because of the ODE. This is the reason for setting $\mathbf{a}_{\theta}(s_{N}) = 0$.
2. In the case of $\mathbf{z}(s_{N})$ since the loss is directly dependent on $\mathbf{z}$, we do not set $\mathbf{a}(s_{N})$ to be $0$.
3. The time parameters $t_{0}$ and $t_{N}$ are made trainable. Hence the paper computes $\frac{dL}{dt_{0}}$ and $\frac{dL}{dt_{N}}$.

We need $\frac{\partial L}{\partial \theta}$ to update the weights of the ODE network, $\frac{dL}{d\mathbf{z}(t_{0})}$ is required to compute the gradients of the part before the ODE. For example in the case of the MNIST model described which is `Inputs -> Downsampling Layers -> ODE Solver -> Classification Layer`, we need $\frac{dL}{d\mathbf{z}(s_{0})}$ to compute the gradients of the downsampling layers.

Equations $(11)$, $(16)$, $(17)$ and $(18)$ provide the gradients for all key components of the ODESolver -- i.e. gradient wrt input, network $f$, and the time parameters. Below is the algorithm for the reverse mode derivative of an ODE.

$$
\begin{array}{l l}
\hline \textbf{Algorithm } \text{Complete reverse-mode derivative of an ODE initial value problem} & \\ \hline \textbf{Input: } \text{model parameters } \theta,\; t_{0},\; t_{1},\; \mathbf{z}(t_{1}),\; \frac{\partial L}{\partial \mathbf{z}(t_{1})} & \\[8pt] \textbf{Output: } \frac{\partial L}{\partial \mathbf{z}(t_0)},\; \frac{\partial L}{\partial \theta},\; \frac{\partial L}{\partial t_0},\; \frac{\partial L}{\partial t_1} & \\[12pt] \frac{\partial L}{\partial t_1} = \frac{\partial L}{\partial \mathbf{z}(t_1)}^\top f(\mathbf{z}(t_1), t_1, \theta) & \triangleright \textit{ Compute gradient w.r.t } t_{1}\\[8pt] s_0 = \left[\mathbf{z}(t_1),\; \frac{\partial L}{\partial \mathbf{z}(t_1)},\; \mathbf{0}_{|\theta|},\; -\frac{\partial L}{\partial t_1}\right] & \triangleright \textit{ Define initial augmented state}\\[8pt] \textbf{def } \text{aug\_dynamics}([\mathbf{z}(t), \mathbf{a}(t), \cdot, \cdot],\; t,\; \theta)\text{:} & \triangleright \textit{ Define dynamics on augmented state}\\[8pt] \quad \textbf{return } \left[f,\; -\mathbf{a}^\top \frac{\partial f}{\partial \mathbf{z}},\; -\mathbf{a}^\top \frac{\partial f}{\partial \theta},\; -\mathbf{a}^\top \frac{\partial f}{\partial t}\right] & \triangleright \textit{ Compute vector-Jacobian products}\\[8pt] \left[\mathbf{z}(t_0),\; \frac{\partial L}{\partial \mathbf{z}(t_0)},\; \frac{\partial L}{\partial \theta},\; \frac{\partial L}{\partial t_0}\right] = \text{ODESolve}(s_0, \text{aug\_dynamics}, t_1, t_0, \theta) & \triangleright \textit{ Solve reverse-time ODE}\\[8pt] \textbf{return } \frac{\partial L}{\partial \mathbf{z}(t_0)},\; \frac{\partial L}{\partial \theta},\; \frac{\partial L}{\partial t_0},\; \frac{\partial L}{\partial t_1} & \triangleright \textit{ Return all gradients}\\[4pt] \hline
\end{array}
$$