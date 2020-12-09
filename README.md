# cacovid
California COVID-19 hospitalization forecasting

Please find the formatted predictions and visualizations in [dumps](dumps).

## Method
We employ a purely data-driven model named ACTS to forecast COVID-19 related data, e.g. confirmed cases, hospitalizations and deaths, along time. We assume that the development of the pandemic in the current region will be highly similar to another region with similar patterns a few months ago. We use attention mechanism to compare and match such patterns and generate forecasts. We also leverage additional features such as demographic data and medical resources to more precisely measure the similarity between regions.

![arch](figs/architecture.png)

For more details about our methodology, previous forecasts and comparison with other models, please refer to our [manuscript](https://arxiv.org/abs/2010.13006) on Arxiv.

## Project Homepage
https://sites.cs.ucsb.edu/~xyan/covid19_ts.html