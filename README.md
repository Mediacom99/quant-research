
# Table of Contents

1.  [Long-only trading portfolio across five equity indices](#org2a27558)
    1.  [Dataset](#org7e77a7e)
    2.  [Task requirements and evaluation](#org5da516d)
2.  [Python versions and dependencies:](#orge298c43)
3.  [Code structure and how to use](#org23aa0d2)
4.  [Useful links, tips and notes:](#org7ddc4c0)


<a id="org2a27558"></a>

# Long-only trading portfolio across five equity indices

The exercise entails the development of a long-only trading model, the output of which will be a portfolio weight matrix across five equity indices.
Each candidate will be provided with a dataset containing:


<a id="org7e77a7e"></a>

## Dataset

-   Five historical series of daily logarithmic returns pertaining to stock indices from five different geographic regions.
-   Six historical series of Fundamental Indicators for each stock index.
-   Two historical series of daily logarithmic returns related to interest rates for each geographic area.
-   Two historical series of Macroeconomic Indicators for each geographic area.
-   Three historical series of daily logarithmic returns related to currency crosses.
-   Three historical series of daily logarithmic returns related to commodities.
-   The resulting portfolios should exclusively comprise a combination of weights of these five stock indices.

The data is available in **data/data.xlsx**


<a id="org5da516d"></a>

## Task requirements and evaluation

The in-sample period spans from September 30, 2003, to December 31, 2019.

Out-of-sample results and the scientific rigor of the model will be the primary criteria for evaluation.

Materials that enhance clarity, comprehensibility, and readability of the model will be highly regarded.

This includes, but is not limited to, the formalization of the model using LaTeX or similar typesetting systems,
detailed and clear code comments, and self-explanatory variable nomenclature to the greatest extent possible.

The model must be implemented using Python, and candidates may employ any libraries or tools at their disposal.


<a id="orge298c43"></a>

# Python versions and dependencies:

1.  **pandas** & **openpyxl** (main data management module using pandas.DataFrame)
2.  **matplotlib** (for plotting stuff)
3.  **numpy** (linear algebra calculations)
4.  **scipy** (optimization algorithms)
5.  **scikit-learn** (linear regression, PCA, model selection through scoring)

Python versions: *python3.11* or *python3.12*


<a id="org23aa0d2"></a>

# Code structure and how to use

Basically just run **run.py** and choose/uncomment out what you dont need.


<a id="org7ddc4c0"></a>

# Useful links, tips and notes:

1.  install python module on Windows: python3.11.exe -m pip install *module-name*
2.  Scikit-learn docs: <https://scikit-learn.org/stable/>
3.  something: <https://www.math.hkust.edu.hk/~makchen/MAFS5140/Chap3>.
4.  huge machine learning library: <https://scikit-learn.org/stable/>
5.  this one is good too: <https://www.statsmodels.org/stable/index.html>

