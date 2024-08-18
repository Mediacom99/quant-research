
# Table of Contents

1.  [Long-only trading portfolio across five equity indices](#org1c1ab6c)
    1.  [Dataset](#orgcb67002)
    2.  [Task requirements and evaluation](#org54dcd22)
2.  [Python versions and dependencies:](#org1ed2822)
3.  [Useful links, tips and notes:](#org055b8c4)


<a id="org1c1ab6c"></a>

# Long-only trading portfolio across five equity indices

The exercise entails the development of a long-only trading model, the output of which will be a portfolio weight matrix across five equity indices.
Each candidate will be provided with a dataset containing:


<a id="orgcb67002"></a>

## Dataset

-   Five historical series of daily logarithmic returns pertaining to stock indices from five different geographic regions.
-   Six historical series of Fundamental Indicators for each stock index.
-   Two historical series of daily logarithmic returns related to interest rates for each geographic area.
-   Two historical series of Macroeconomic Indicators for each geographic area.
-   Three historical series of daily logarithmic returns related to currency crosses.
-   Three historical series of daily logarithmic returns related to commodities.
-   The resulting portfolios should exclusively comprise a combination of weights of these five stock indices.

The data is available in **data/data.xlsx**


<a id="org54dcd22"></a>

## Task requirements and evaluation

The in-sample period spans from September 30, 2003, to December 31, 2019.

Out-of-sample results and the scientific rigor of the model will be the primary criteria for evaluation.

Materials that enhance clarity, comprehensibility, and readability of the model will be highly regarded. This includes,
but is not limited to, the formalization of the model using LaTeX or similar typesetting systems, detailed and clear code comments,
and self-explanatory variable nomenclature to the greatest extent possible.

The model must be implemented using Python, and candidates may employ any libraries or tools at their disposal.


<a id="org1ed2822"></a>

# Python versions and dependencies:

1.  pandas & openpyxl (main data management module using pandas.DataFrame)
2.  matplotlib (for plotting stuff)
3.  numpy (linear algebra calculations)
4.  scipy (optimization algorithms)
5.  scikit-learn (linear regression, PCA, model selection through scoring)

Python versions: *python3.11* or *python3.12*


<a id="org055b8c4"></a>

# Useful links, tips and notes:

1.  install python module on Windows: python3.11.exe -m pip install *module-name*
2.  something: <https://www.math.hkust.edu.hk/~makchen/MAFS5140/Chap3>.
3.  huge machine learning library: <https://scikit-learn.org/stable/>
4.  this one is good too: <https://www.statsmodels.org/stable/index.html>

