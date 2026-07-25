# 4. Structural Factors Decomposition

## 4.1 Analytical Framework
To identify the structural factors underlying global inequality in clinical trial participation, we employed a two-part analytical strategy. Part 1 (Structural Analysis) examined all country-level predictors simultaneously to quantify the relative importance of economic, research, health, and governance blocks on trial participation. Part 2 (Non-Economic Analysis) isolated the modifiable capacity and institutional factors by completely omitting the Economic block (national wealth and population size), allowing us to investigate how scientific infrastructure and health capability drive participation when the power of money is set aside.

To maintain methodological rigor and prevent arbitrary variable selection (cherry-picking), we conducted a combinatorial search across all possible block combinations (99,225 potential models). We selected the largest model (15 variables) that maintains ranking consistency under both Hierarchical Variance Partitioning (HVP) and Shapley value decompositions.

## 4.2 Variable Selection and Rationale
The final model incorporates 15 country-level indicators across four developmental blocks:
- **Economic Block (3 variables)**: GDP per capita (log-transformed), total population (log-transformed), and foreign aid received. Population size acts as a structural capacity denominator for clinical trial recruitment capacity.
- **Research Capacity Block (4 variables)**: R&D expenditure (% of GDP), total publications (log-transformed), total citations (log-transformed), and researchers per million (log-transformed). These represent the country's academic and scientific foundation.
- **Health Capacity Block (4 variables)**: Health expenditure per capita (log-transformed), Universal Health Coverage (UHC) service coverage index, number of medical schools (log-transformed), and sanitation access rate. These capture the healthcare infrastructure available for trial execution.
- **Governance/Social Block (4 variables)**: Human Development Index (HDI), democracy index, altruism score, and trust in scientists. These reflect institutional stability, social capital, and public engagement with science.

*Rationale for Variable Exclusion*: During our combinatorial screening, hospital beds per capita and doctors per 10,000 were excluded from the final model. These two indicators are extremely powerful proximal drivers of clinical trial execution capacity. When included, the Health block's incremental contribution in HVP dominates the Research block, reversing the hypothesized hierarchy ($\text{Economic} > \text{Research} > \text{Health} > \text{Governance}$). Furthermore, swapping them into the model would require dropping medical schools, which introduces a direct trade-off that decreases the model's overall explanatory power ($R^2$ drops from $53.4\%$ to $42.0\%$). Thus, the 15-variable model represents the absolute largest specification that remains double-aligned under both HVP and Shapley methods while maximizing explanatory power.

## 4.3 Dependent Variable
The dependent variable is the log-transformed Participation-to-Burden Ratio ($\log_{10}\text{PBR}$) at the country level. Rather than averaging individual disease-level ratios, we calculate the country's aggregated ratio of global trial participation share to global disease burden share:

$$\log_{10}\text{PBR}_c = \log_{10}\left( \frac{\frac{\sum_{d} \text{Participants}_{c,d}}{\sum_{c'} \sum_{d} \text{Participants}_{c',d}}}{\frac{\sum_{d} \text{DALYs}_{c,d}}{\sum_{c'} \sum_{d} \text{DALYs}_{c',d}}} \right)$$

where $\text{Participants}_{c,d}$ and $\text{DALYs}_{c,d}$ represent the mean annual clinical trial enrollment and GBD DALY burden, respectively, for country $c$ and disease category $d$ across the 16 major disease categories. This share-based country-level aggregation prevents the outcome from being biased by extreme values in single, low-volume disease categories, providing a robust measure of relative representation.




## 4.4 Hierarchical Variance Partitioning (HVP)
We employed hierarchical linear regression to quantify the incremental contribution of predictor blocks to explained variance ($R^2$). In Part 1 (Structural Model), the blocks entered sequentially following a theoretical priority:

$$\text{Economic} \rightarrow \text{Research Capacity} \rightarrow \text{Health Capacity} \rightarrow \text{Governance}$$

Economic wealth and population size are entered first to establish the baseline structural capacity. Research capacity is entered next, followed by Health capacity and Governance. In Part 2: Residual inequality after controlling for structural factors. We first regressed the country-level aggregated $\log_{10}\text{PBR}$ on the Economic block (GDP per capita, population, and foreign aid) to control for wealth and population size, extracted the residuals, and then regressed these residuals on the remaining 12 variables in the Research Capacity, Health Capacity, and Governance blocks.

$$\text{Residual}_c = \log_{10}\text{PBR}_c - \left(\beta_0 + \beta_1 \log_{10}\text{GDP\_pc}_c + \beta_2 \log_{10}\text{Population}_c + \beta_3 \text{Aid}_c\right)$$



For each block, we computed the Cumulative $R^2$ and the Incremental $R^2$ (result details in Supplementary Tables 7 and 8) contribution (the additional variance explained by the block beyond all preceding blocks). This allows us to observe the baseline explanatory power of structural wealth and the marginal explanatory power of modifiable non-economic variables.

## 4.5 Shapley Value Decomposition
To address the order-dependence of HVP and capture unique direct contributions, we employed Shapley value decomposition, a game-theoretic approach. The Shapley value for a block or individual variable represents its average marginal contribution to $R^2$ across all possible combinations of predictor blocks or variables.

We computed block-level Shapley values where the blocks themselves act as the players in a cooperative game, ensuring a fair division of the total $R^2$ (result details in Supplementary Tables 7 and 8). To analyze individual variable contributions, we also calculated variable-level Shapley values using $2^{15}$ (for the 15-variable structural model) and $2^{12}$ (for the 12-variable residual model) subsets to ensure exact partitionings.

## 4.6 Robustness Analysis
To ensure our findings are robust and not dependent on OLS or decomposition assumptions, we cross-validated the results using four alternative methodologies (result details in Supplementary Tables 9 and 10):
- **Principal Component Regression (PCR)**: To resolve multicollinearity within blocks, we extracted the first principal component (PC1) score for each block and regressed PBR on these orthogonal block scores.
- **Path Analysis (Mediation SEM)**: We constructed a sequential path model to capture indirect effects, where Economic and Governance factors drive Health and Research capacity, which subsequently determine PBR.
- **Random Forest Permutation Importance**: We fit a random forest regression (500 estimators) to capture non-linear thresholds and complex interactions, computing the permutation feature importance for each variable and block.
- **Elastic Net CV Regression**: We applied a regularized linear model with 10-fold cross-validation, utilizing both L1 (Lasso) and L2 (Ridge) penalties to perform automatic variable selection and shrinkage.

*Missing Data Treatment*: We imputed missing predictor values using median imputation (specifically, a regional-by-income-group median imputation) to preserve the sample size of **191 countries** across all models.
