## Goals

Much of the literature on risk-preference focuses on the costs/benefits to the individual of being risk-seeking/risk-averse. But if we think of intergroup or interspecies competition, it seems that a group with more diversity will beat a group with less diversity in a dynamic environment, as long as there is resource/information sharing within the group.
We would want to set up the simplest environment possible that would allow us to test this.
Brainstorming: two groups, each with 30 people. 50 possible foraging sites. Each day, each person in each group can visit 1 foraging site. When they return home each night, the groups pool and share the rewards. If both groups visit the same site, the group with more members will get the rewards. (e.g. two people from group A visit site 1 and only 1 person from group B visits site 1, so group A get all the rewards) if it is a tie, then it goes by distance to the home site of the group.
Once we have the basic environment set up we can look at diversity in agent parameters: learning rate, exploration/exploitation tradeoff, utility function, delay discounting, etc

## TODO

- [X] Feb 18. Skim [Julia Manual](https://docs.julialang.org/en/v1/manual/) and first two chapters of *Reinforcement Learning*.
- [X] Feb 25. Learn [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/).
- [ ] Mar 4. Implement Simple Environment.
- [ ] Mar 11. Implement Complex Environments.
- [ ] [HPC application](https://nyu.service-now.com/servicelink/catalog.do?sysparm_document_key=sc_cat_item,b0fc230be498d6408b4d97a033492665)

## Simple Environment

+ Gridspace((100, 100))
+ 30 patches, random positions
+ 2 forager groups, 30 foragers per group. (Something we can play with.) Forager/patch ratio is an important parameter.

### Forager

1. Learning rate (same for self and vicarious) $`\alpha`$. $`\log(\alpha) \sim \mathscr{N} (\mu_{\log\alpha}, \sigma_{\log\alpha})`$
2. Preferences $`k, \rho`$. $`\rho \sim \mathscr{U} (0.6, 1.3)`$, $`k \sim \mathscr{U} (0.001, 1), (k < 1)`$
    Utility $`U = \frac{V^\rho}{1 + kd}`$ where d is euclidean distance.
3. Softmax temperature $`\beta`$, $`\beta \sim \mathscr{U} (0.1 10)`$
4. Q. $`\Delta Q = \alpha(U - Q)`$

#### agent_step!

1. Pick a patch (based on softmax of Q)
2. Increment patch.visit_counts[gid]

### Patch

1. $`\mu`$ draws from a Gaussian and is a Gaussian Random Walk with $`\sigma_\text{walk}`$ (for some simulations we would have low sigma, others high sigma. If we want to get fancy we can do something like each patch has its own $`\sigma_\text{walk}`$, or use hidden-markov-model for patch quality)
2. reward $`V \sim \mathscr{N}(\mu, \sigma)`$

#### agent_step!

1. Sample a reward
2. Count foragers from each group. ~if A > B give food to A otherwise B. If A==B flip a coin~pick a group to give reward based on softmax of visit_counts
3. Update Q of self and others. (there is learning of "crowdedness" of patches)

Panel A: rew rate v time group by group with thin line individuals
Panel B: normilize against group with perfect information (top n patches)

\rho = 1 risk neutral, different sigma, plot log(CRA/CRB) as heatmap

min(group A)

## Complex Environment
Death and reproduction
+ Foragers wander around a grid world to find patches.
+ They have some sleep drive. So in the morning they head out and then they need to return home to sleep.
+ If they don't make it home before dark there is a chance of death.
+ They can die if they don't get enough food. Food is pooled in the group
+ Each group has a food store, so if there is surplus food they can save it. Food has a decay rate.
+ If food storage crosses some threshold, can create new foragers.
+ New property: age. Babies (age< 21 days) eat but don't forage.  Probabilty of death increases with age?
