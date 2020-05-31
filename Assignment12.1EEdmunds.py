# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:32:23 2020

@author: Evan
"""
import pandas
import numpy as np
import thinkstats2
import thinkplot
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as smf

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)


df = pandas.read_csv("crime.csv")
df['month'] = pandas.to_datetime(df['month'], dayfirst = True)
    
def all_cdf(x):
    for i in x:
        cdf = thinkstats2.Cdf(df[i], label = i)
        thinkplot.Cdf(cdf)
        thinkplot.Show(title = i, xlabel = 'Crime Count', ylabel = "CDF")

def all_hist(x):
    for i in x:
        _ = plt.hist(df[i], bins = 20)
        _ = plt.title(i)
        _ = plt.xlabel('Number of Crimes')
        _ = plt.ylabel('Count (months)')
        plt.show()

def summ_stats(x):
    for i in x:
        base = thinkstats2.Pmf(df[i])
        mean = base.Mean()
        mode = base.Mode()
        spread = base.Var()
        tails = df[i].kurtosis()
        print("{} Crime Stats: mean = {:.2f}, mode = {:.2f}, spread = {:.2f}, tails = {:.2f}."
              .format(i, mean, mode, spread, tails))

def ShowPMF(pmf1, pmf2):
    thinkplot.PrePlot(num = 2)
    thinkplot.Pmfs([pmf1,pmf2])
    thinkplot.Show(title = 'Overall and Recent Crime Patterns', 
                   xlabel = 'Crime Count', ylabel = 'PMF')

def MakeNormalPlot(x):
    """Generates a normal probability plot of birth weights."""

    mean, var = thinkstats2.TrimmedMeanVar(df[x], p=0.01)
    std = math.sqrt(var)

    xs = [-4, 4]
    fxs, fys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(fxs, fys, linewidth=4, color='0.8')

    thinkplot.PrePlot(2) 
    xs, ys = thinkstats2.NormalProbability(df[x])
    thinkplot.Plot(xs, ys, label='Number of Crimes')
    thinkplot.Show(title = 'Normal Prob Plot: {}'.format(x),
                   xlabel='Standard deviations from mean',
                   ylabel='Number of Crimes')

def scatter(x):
    tot_crimes = df.Total_crimes
    thinkplot.Scatter(df[x], tot_crimes, alpha = .5)
    if x == 'month':
        thinkplot.Show(title = "Total Crimes vs Time",
                       xlabel = "Year",
                       ylabel = "Total Crimes")
    else:
        thinkplot.Show(title = "Total Crimes vs " + x + " Crimes",
                       xlabel = x + " Crimes",
                       ylabel = "Total Crimes")
        print(x + " crime stats")
        print("Spearman's correlation:", thinkstats2.SpearmanCorr(tot_crimes, df[x]))
        print("Covariance:", thinkstats2.Cov(tot_crimes, df[x]))
        print()

class CorrelationPermute(thinkstats2.HypothesisTest):
    
    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.SpearmanCorr(xs, ys))
        return test_stat
    
    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

def corr_test():
    data = df['Theft'], df['Serious']
    ht = CorrelationPermute(data)
    pvalue = ht.PValue()
    print(pvalue)

def SummarizeResults(results):
    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print('%s   %0.3g   (%.3g)' % (name, param, pvalue))

    try:
        print('R^2 %.4g' % results.rsquared)
        ys = results.model.endog
        print('Std(ys) %.4g' % ys.std())
        print('Std(res) %.4g' % results.resid.std())
    except AttributeError:
        print('R^2 %.4g' % results.prsquared)

def Regress(x):
    formula = 'Total_crimes ~ ' + x
    model = smf.ols(formula, data=df)
    results = model.fit()
    print(x + ' regression analysis')
    SummarizeResults(results)
    print()
    
# CDFs of all variables
all_cdf(df.columns[1:])

# Histograms of all variables
all_hist(df.columns[1:])

# Summary Stats of all variables
summ_stats(df.columns[1:])

# Generating PMFs for Total crimes of all times and past 5 years
first_pmf = thinkstats2.Pmf(df.Total_crimes, label="PMF Crimes (2003-2020)")
second_pmf = thinkstats2.Pmf(df.Total_crimes[-60:,], label = "PMF Crimes Last 5 Years")

# Plotting the PMFs
ShowPMF(first_pmf, second_pmf)

# Normal Probability Plots for variables
MakeNormalPlot('Hooligan')
MakeNormalPlot('Drugs')

# Variables for the Scatterplot
scatter('Serious')
scatter('Theft')
scatter('month')

# Correlation Matrix for all variables
print(df.corr(method = 'spearman'))

# Testing the p-value for correlation
print(thinkstats2.SpearmanCorr(df.Theft, df.Serious))
corr_test()

# Creating a Regression Model
Regress('Theft')
Regress('Serious')
