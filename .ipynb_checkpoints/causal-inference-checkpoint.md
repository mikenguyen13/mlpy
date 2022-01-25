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