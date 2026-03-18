### Variance Decomposition

- $ N $ = Number of Companies or Sectors
- $M$ = Number of Members per Company or per Sector
- $\bar{x}_j$ = Mean of group $j$
- $\bar{x}$ = Grand mean

------

**Start** by adding and subtracting the group mean inside the total variance:

$$ \frac{1}{NM}\sum_j\sum_i\left(x_{ij}  - \bar{x}_j + \bar{x}_j - \bar{x}\right)^2 $$

**Expand** the square — letting $a = (x_{ij} - \bar{x}_j)$ and $b = (\bar{x}_j - \bar{x})$:

$$ = \frac{1}{NM}\sum_j\sum_i\left[(x_{ij} - \bar{x}*j)^2 + 2(x*{ij} - \bar{x}_j)(\bar{x}_j - \bar{x}) + (\bar{x}_j - \bar{x})^2\right] $$

**Cross term vanishes.** For any fixed $j$, the factor $(\bar{x}_j - \bar{x})$ is constant across $i$, so:

$$ \sum_i (x_{ij} - \bar{x}_j)(\bar{x}_j - \bar{x}) = (\bar{x}*j - \bar{x})\underbrace{\sum_i(x*{ij} - \bar{x}*j)}*{=;0} = 0 $$

This leaves two terms.

------

**Term 1 — Within-Group Variance:**

$$ \frac{1}{NM}\sum_j\sum_i(x_{ij} - \bar{x}_j)^2 = \frac{1}{NM}\sum_j M\sigma_j^2 = \frac{1}{N}\sum_j \sigma_j^2 $$

This is the **average of the group variances**.

**Term 2 — Between-Group Variance:**

$$ \frac{1}{NM}\sum_j\sum_i(\bar{x}_j - \bar{x})^2 = \frac{1}{NM}\sum_j M(\bar{x}_j - \bar{x})^2 = \frac{1}{N}\sum_j(\bar{x}_j - \bar{x})^2 $$

This is the **variance of the group means around the grand mean**.

------

**Result:**
$$
\boxed{
\underbrace{\frac{1}{NM}\sum_j\sum_i(x_{ij} - \bar{x})^2}_{\text{Total Variance}}
=
\underbrace{\frac{1}{N}\sum_j \sigma_j^2}_{\text{Within-Group}}
+
\underbrace{\frac{1}{N}\sum_j(\bar{x}_j - \bar{x})^2}_{\text{Between-Group}}
}
$$

Total variance decomposes **exactly** into the average within-group variance plus the variance of the group means. No approximation, no assumptions on the $\sigma_j^2$ — the cross term is zero by construction.

### Variance Decomposition — Unequal Group Sizes

- $ N $ = Number of groups
- $M_j$ = Number of members in group $j$, with $\sum_j M_j = M$
- $\bar{x}_j = \frac{1}{M_j}\sum_i x_{ij}$ = mean of group $j$
- $\bar{x} = \frac{1}{M}\sum_j\sum_i x_{ij} = \frac{1}{M}\sum_j M_j\bar{x}_j$ = grand mean (now a **weighted** average of group means)
- $\sigma_j^2 = \frac{1}{M_j}\sum_i(x_{ij} - \bar{x}_j)^2$ = variance of group $j$

------

**Start** with total variance, normalized by $M$ (total observations):

$$ \frac{1}{M}\sum_j\sum_i\left(x_{ij} - \bar{x}\right)^2 $$

**Add and subtract** the group mean inside the square:

$$ = \frac{1}{M}\sum_j\sum_i\left(x_{ij} - \bar{x}_j + \bar{x}_j - \bar{x}\right)^2 $$

**Expand:**

$$ = \frac{1}{M}\sum_j\sum_i\left[(x_{ij} - \bar{x}_j)^2 + 2(x_{ij} - \bar{x}_j)(\bar{x}_j - \bar{x}) + (\bar{x}_j - \bar{x})^2\right] $$

**Cross term vanishes.** For fixed $j$, $(\bar{x}_j - \bar{x})$ is constant across $i$:

$$ \sum_i (x_{ij} - \bar{x}_j)(\bar{x}_j - \bar{x}) = (\bar{x}*j - \bar{x})\underbrace{\sum_i(x*{ij} - \bar{x}*j)}_{= 0} = 0 $$

This holds regardless of group size — the inner sum is zero by definition of $\bar{x}_j$.

------

*Term 1 — Within-Group Variance:**
$$
\frac{1}{M}\sum_j\sum_i(x_{ij} - \bar{x}_j)^2 = \frac{1}{M}\sum_j M_j\sigma_j^2 = \sum_j \frac{M_j}{M}\sigma_j^2 
$$
A **weighted average** of group variances, with weights $w_j = M_j/M$.

**Term 2 — Between-Group Variance:**

$$\large \frac{1}{M}\sum_j\sum_i(\bar{x}_j - \bar{x})^2 = \frac{1}{M}\sum_j M_j(\bar{x}_j - \bar{x})^2 = \sum_j \frac{M_j}{M}(\bar{x}_j - \bar{x})^2 $$

A **weighted variance** of the group means around the grand mean, with the same weights $w_j = M_j/M$.

------

**Result:**
$$
\boxed{ \underbrace{\frac{1}{M}\sum_j\sum_i(x_{ij} - \bar{x})^2}_{\text{Total Variance}}
=
\underbrace{\sum_j \frac{M_j}{M}\sigma_j^2}_{\text{Within-Group}} 
+ 
\underbrace{\sum_j \frac{M_j}{M}(\bar{x}_j - \bar{x})^2}_{\text{Between-Group}} } 
$$


------

**Comparison to Equal Group Sizes**

When all $M_j = M/N$, the weights $M_j/M$ reduce to $1/N$, and the grand mean $\bar{x} = \frac{1}{M}\sum_j M_j\bar{x}_j$ reduces to $\frac{1}{N}\sum_j \bar{x}_j$, recovering the equal-size result exactly.

The structure is identical — the only change is that **simple averages become size-weighted averages**. Larger groups pull both the grand mean and both variance terms toward their values. This is consequential: a group with $M_j \approx M$ can dominate the decomposition even if its internal variance is small.
