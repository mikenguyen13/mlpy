# Causal Inference 

Three rungs of causal inference

 1. Seeing: Correlations (Basic Stat, Machine Learning)
     * Correlation can be misinterpreted (i.e., spurious correlations), which might lead to Simpson's paradox
 2. Doing: Causal Inference by cohorts by intervention
     * Controlling for confounding variables, scientists can infer causality
 3. Imagining: Causal Inference for individuals by counterfactuals
     * Can answer What-if questions. 

 * Limitations of Correlations 
 * Simpson's Paradox 
 * Graph Models

[Causation Assumptions](https://www.youtube.com/watch?v=w4DFQmh5OGw&ab_channel=PyData): 

 * Stable Unit Treatment value Assumption (SUTVA) (i.e., individuals do not impact each other)
 * Consistency - a **potential** outcome is the same as an **actual** outcome 
 * Positivity - every cohort has a non-zero probability 
 * Ignorability - "No unmeasured confounders" assumption (i.e., no alternative explanation). 
 
 
## Counter Factual Analysis for Explainble AI (XAI) (https://www.youtube.com/watch?v=DgzyKrLxIaU&ab_channel=PyData)

A counterfactual explanation of a prediction describes the smallest change to the feature values that changes the prediction to a predefined output 

Actionable Counterfactuals: those values that can be changed
Desirable Properties 
1. Interpretable and human friendly explanations
2. Small number of feature changes
3. Avoid contradictions 
4. Produce the predefined prediction 
5. Similar to the instance regarding feature values 
6. Generate multiple counterfactual explanations
7. Should have feature values that are likely 

Optimization Challenges

1. Prediction of counterfactual should be as close to desired prediction

$$
o_1 (f(x_{cf}), y_{cf}) = 
\begin{cases}
0, & \text{if} f(x_{cf}) \in y_{cf} \\
\inf_{cf}| f(x_{cf} - y_{cf}| & \text{else}
\end{cases}
$$

2. Counterfactual should be as similar as possible to the instance

$$
o_2 (x_0, x_{cf}) = \frac{1}{p} \sum_{j=1}^p \delta_G (x_0^j , x_{cf}^j)
$$

$$
\delta_G(x_0^j, x_{cf}^j) = 
\begin{cases}
\frac{1}{R_j} |x_0^j - x_{cf}^j | & \text{if } x_0^j \text{ is numerical} \\
\Pi_{x_0^j \neq x_{cf}^j} & \text{if } x_0^j \text{ is categorical}
\end{cases}
$$

3. Sparse feature change
$$
o_3(x_0 , x_{cf}) = ||x_0 - x_{cf}||_0 = \sum_{j=1}^p \Pi_{x_0^j \neq x_{cf}^j}
$$

4. Counterfactuals should have likely feature values
$$
o_4 (x_{cf}, X^{obs}) = \frac{1}{p} \sum_{j=1}^p \delta_G(x_{cf}^j, x_{[1]}^j)
$$

Algorithms for Counterfactuals
1. Base method [@wachter2017counterfactual]: Optimizes criterion 1 & 2
2. Multi-objective Counter Factual Explanations: Optimize all criteria
3. Diverse Counterfactual Explanations (DiCE)
    * Generates multiple CFs
    * Optimizes criterion 1 & 2 along with an extra diversity criterion 
4. Interpretable CFs guided by prototypes
    * generates interpretable CFs that lies closer to data distribution an actionable
    * Optimizes criterion 1, 2, and 3
    * can apply to image data

<br>

Diverse CF Explanations (DiCE)
The loss function optimized is 
$$
L = \underset{x_{cf_1}, \dots, x_{cf_k}}{\operatorname{argmin}} \frac{1}{k} \sum_{i=1}^k y \text{loss} (x_{cf}, y_{cf}) + \frac{\lambda_1}{k} \sum_{i=1}^k \text{dist}(x_{cf}, x_0) - \lambda_2 \text{dpp_diverrsitty}(x_{cf_1}, \dots, x_{cf_k})
$$

* $x_{cf_1}$ is a CF example
* k -total no. of CFs to be generated 
* $f(.)$ model 
* $x_0$ original input
* $\lambda_1, \lambda_2$ - hyper parameter
* yloss = $l_1$ or $l_2$ loss
* dist = a distance function 
    * continuous variables: $\text{dist_cont}(x_{cf}, x_0) = \frac{1}{d_{cont}}\sum_{j=1}^{d_{cont}} \frac{|x_{cf}^j - x_0^j|}{MAD_j}$
    * categorical variables: $\text{dist_cat}(x_{cf}, x_0) = \frac{1}{d_{cont}}\sum_{j=1}^{d_{cat}}\Pi(x_{cf}^j \neq x_0^j)$
* dpp_diversity: term based on determinantal point processes to ensure diversity among CFs $dpp_diversity = det(K), K_{i,j}= \frac{1}{1 + dist (x_{cf_i}, x_{cf_j})}$








