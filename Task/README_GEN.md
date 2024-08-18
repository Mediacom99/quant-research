
# Table of Contents

1.  [The exercise entails the development of a long-only trading model, the output of which will be a portfolio weight matrix across five equity indices.](#org239b871)
2.  [TASK](#org1451339)
3.  [Python modules required:](#org91f9c74)
4.  [COSE UTILI:](#org0263245)


<a id="org239b871"></a>

# The exercise entails the development of a long-only trading model, the output of which will be a portfolio weight matrix across five equity indices.


<a id="org1451339"></a>

# TASK

Each candidate will be provided with a dataset containing:

Five historical series of daily logarithmic returns pertaining to stock indices from five different geographic regions.
Six historical series of Fundamental Indicators for each stock index.
Two historical series of daily logarithmic returns related to interest rates for each geographic area.
Two historical series of Macroeconomic Indicators for each geographic area.
Three historical series of daily logarithmic returns related to currency crosses.
Three historical series of daily logarithmic returns related to commodities.
The resulting portfolios should exclusively comprise a combination of weights of these five stock indices.

The in-sample period spans from September 30, 2003, to December 31, 2019.

Out-of-sample results and the scientific rigor of the model will be the primary criteria for evaluation.

Materials that enhance clarity, comprehensibility, and readability of the model will be highly regarded. This includes, but is not limited to, the formalization of the model using LaTeX or similar typesetting systems, detailed and clear code comments, and self-explanatory variable nomenclature to the greatest extent possible.

The model must be implemented using Python, and candidates may employ any libraries or tools at their disposal.


<a id="org91f9c74"></a>

# Python modules required:

1.  pandas & openpyxl (main data management module using pandas.DataFrame)
2.  matplotlib (for plotting stuff)
3.  numpy (linear algebra calculations)
4.  scipy (optimization algorithms)
5.  scikit-learn (linear regression, PCA, model selection through scoring)

Python versions: *python3.11* or *python3.12*


<a id="org0263245"></a>

# COSE UTILI:

1.  install module: python3.10.exe -m pip install <module<sub>name</sub>>
    <https://www.math.hkust.edu.hk/~makchen/MAFS5140/Chap3.pdf>
2.  huge machine learning library: <https://scikit-learn.org/stable/>
3.  or this one is good: <https://www.statsmodels.org/stable/index.html>

