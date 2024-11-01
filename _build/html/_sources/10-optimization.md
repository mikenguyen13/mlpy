# Optimization

## Modeling

$p_x$ dimensional input vector $\mathbf{x}_i$  
associated target vector $\mathbf{y}_i$ for each of the $i = 1, \dots, n$ data points.

We create a model $f$ with a set of $p_\theta$ parameters $\mathbf{\theta}$ as:

$$
f(\mathbf{x}_i; \mathbf{\theta})
$$

Our target vector $\mathbf{y}_i$ can be 

 1. A single continuous output value
 2. Membership in one k classes
 
To find solution to your prediction problems, we have to find $\theta$ 

## Finding $\theta$

To find $\theta$, a loss (objective) function which describes the distance between $\mathbf{y}_i$ and $f(\mathbf{x}_i; \mathbf{\theta})$ must be constructed 
In the first example, we can use the squared error loss

$$
Q(f(\mathbf{x}); y; \mathbf{\theta}) = \sum_{i = 1}^n (y_i - f(\mathbf{x}_i; \mathbf{\theta}))^2
$$
In the second example, we can use Cross Entropy Loss 

$$
Q(f(\mathbf{x}); y ; \mathbf{\theta}) = - \frac{1}{n} \sum_{i=1}^n \sum_{c=1}^{10} y_{i, c} \log (f(\mathbf{x}_i, \mathbf{\theta})_c)
$$
In either case, we seek a set of model parameters that minimize the loss function $Q$

$$
\hat{\mathbf{\theta}} = \underset{\theta}{\operatorname{argmax}}  (Q(f(\mathbf{x}); y; \mathbf{\theta})
$$

A note on classification 

 * Similar to Generalized Linear Models, we must consider predicting a probability $p \in (0,1)$ so we should ensure that whatever is coming out of our model maps to that interval 
 * Typically, we will let models learn on whatever space they what and then squeeze them into $(0,1)$ with the **softmax function**

$$
p(y_i = k |\mathbf{x}_i, \mathbf{\theta}) = \frac{\exp(\mathbf{\theta}_k' \mathbf{x}_i)}{\sum_{l = 1}^K \exp(\mathbf{\theta}_l' \mathbf{x}_i)}
$$
 * This function takes a linear outputs for each class and scales $(0,1)$ so that their sum is 1
 * in the binary case, this would jut be the sigmoid
 
## Finding a suitable $\theta$

* Goal: Find a set of $\mathbf{\theta}$ that minimize the loss function $Q$
* Idea: start with a guess and iterate, using the gradient as a guide 

Steps

$$
\hat{\mathbf{\theta}}^{(j + 1)} = \hat{\mathbf{\theta}}^{(j)} - \epsilon_j \mathbf{A}_j \frac{\partial Q (\hat{\mathbf{A}}^{(j)} ) }{\partial \mathbf{\theta}}
$$
where
 * $\frac{\partial Q (\hat{\mathbf{A}}^{(j)} ) }{\partial \mathbf{\theta}}$ the gradient of the objective function $Q(\mathbf{\theta})$ at $\hat{\mathbf{\theta}}$
     * Tells us direction each parameter should move to reduce loss 
     * Also denoted as $\nabla_\theta Q (\hat{\mathbf{\theta}}^{(j)})$
 * $\epsilon_j$ is the learning rate, which alows us to stretch or shrink the magnitude of movement 
 * $\mathbf{A}_j$ is a positive definite matrix 
     * Typically the inverse of the squared Jacobian (Gauss-Newton) or Hessian (Newton-Raphson)
     * Allows the optimization to have a deeper understanding of the surface being optimized. 
     * Basically, you incorporate more information so that you can reach your goal faster