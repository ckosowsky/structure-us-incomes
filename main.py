##
## This is file `main.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `main')
## 
## 
## 
## Copyright 2024 Conrad Kosowsky
## 
## ********************************************************
## *        DO NOT MODIFY THE FOLLOWING NOTICE            *
## *                                                      *
## * This file contains (un)modified material             *
## * from the Github repository "structure-us-incomes,"   *
## * which contains replication files for "The Structure  *
## * of the U.S. Income Distribution" by Conrad Kosowsky. *
## * The repository is available at                       *
## *                                                      *
## * https://github.com/ckosowsky/structure-us-incomes    *
## *                                                      *
## * This file is distributed under a modified version    *
## * of the GNU General Public License v. 3.0. An         *
## * unmodified version of the License is available at    *
## *                                                      *
## * https://www.gnu.org/licenses/gpl-3.0.txt             *
## *                                                      *
## * All modifications are non-permissive additional      *
## * term added pursuant to section 7 of the GNU General  *
## * Public License. The additional terms are as follows: *
## *                                                      *
## * 1. If you propagate a modified version of this       *
## *    Program, you must include an unmodified copy      *
## *    of this notice displayed prominently in your      *
## *    software.                                         *
## *                                                      *
## * 2. You may not suggest that Conrad Kosowsky endorses *
## *    any product made with this Program unless you     *
## *    have written permission from Conrad Kosowsky to   *
## *    do so.                                            *
## *                                                      *
## * Further, if you use this Program for any purpose,    *
## * you are encouraged (but not required) to cite "The   *
## * Structure of the U.S. Income Distribution,"          *
## * assuming it is appropriate to do so.                 *
## *                                                      *
## *                   END OF NOTICE                      *
## ********************************************************
## 
## PLEASE KNOW THAT THIS FREE SOFTWARE IS PROVIDED TO
## YOU WITHOUT ANY WARRANTY AND WITHOUT ANY LIABILITY
## FOR THE CREATOR OR DISTRIBUTOR. See sections 15 and
## 16 of the GNU General Public License for more
## information.
## 
## 
## 
##### Switches! #####
do_load_data = False
do_2023_short = False
do_2023_long = False
do_Fisk = False
do_InvG = False
do_CS_InvG = False    # <--- present for historical reasons
do_CSS_InvG = False
do_bootstrap = False
do_figures = False
do_test = False
#####################
if not (do_load_data
        or do_2023_short
        or do_2023_long
        or do_InvG
        or do_Fisk
        or do_CS_InvG
        or do_CSS_InvG
        or do_bootstrap
        or do_figures
        or do_test):
  print()
  print("Warning. This run of estimate_parameters will not do anything.")
  print("To enable data analysis, set one or more do_ booleans to True.")
  print()
  quit()
import bin
import bootstrap
import check_constants as cc
import estimate_parameters as est
import matplotlib as mpl
import matplotlib.pyplot as plt
import make_figures
import numpy as np
import pandas as pd
import scipy.special as spec
import time
dist = ["GB2", "Dagum", "Burr", "Fisk", "InvG", "CS_InvG", "CSS_InvG",
  "Davis", "LogN_P_cut", "LogN_P_mix"]
dist_names = {
  "GB2": "Generalized Beta, type II",
  "Dagum": "Dagum",
  "Burr": "Burr (Singh-Maddala)",
  "Fisk": "Fisk",
  "InvG": "Inverse Gamma",
  "CS_InvG": "Constant-Shift Inverse Gamma",
  "CSS_InvG": "Constant-Shift-Scale Inverse Gamma",
  "Davis": "Davis",
  "LogN_P_cut": "Log-Normal, Pareto Cutoff",
  "LogN_P_mix": "Log-Normal, Pareto Mixture"}
years = [i for i in range(1967, 2024) if (i != 1970)]
data = {}
ecdfs = {}
def the_time():
  print("The time is", time.asctime())
dat_max_min = {
  1967:(-4000,60000),
  1968:(-4000,90000),
  1969:(-4000,90000),
  1971:(-9000,90000),
  1972:(-9000,90000),
  1973:(-9000,90000),
  1974:(-9000,90000),
  1975:(-9000,90000),
  1976:(-9000,90000),
  1977:(-9000,100000),
  1978:(-9000,100000),
  1979:(-9000,100000),
  1980:(-9000,100000),
  1981:(-9000,100000),
  1982:(-9000,120000),
  1983:(-9000,120000),
  1984:(-9000,120000),
  1985:(-9000,200000),
  1986:(-9000,150000),
  1987:(-9000,200000),
  1988:(-9000,200000),
  1989:(-9000,200000),
  1990:(-9000,200000),
  1991:(-9000,200000),
  1992:(-9000,200000),
  1993:(-9000,200000),
  1994:(-9000,250000),
  1995:(-9000,250000),
  1996:(-8000,450000),
  1997:(-4000,450000),
  1998:(-9000,450000),
  1999:(-9000,450000),
  2000:(-9000,350000),
  2001:(-9000,450000),
  2002:(-11000,450000),
  2003:(-11000,550000),
  2004:(-11000,550000),
  2005:(-11000,650000),
  2006:(-11000,550000),
  2007:(-9000,650000),
  2008:(-9000,650000),
  2009:(-9000,550000),
  2010:(-9000,550000),
  2011:(-9000,1000000),
  2012:(-9000,1000000),
  2013:(-9000,1000000),
  2014:(-9000,1000000),
  2015:(-9000,1000000),
  2016:(-9000,1000000),
  2017:(-9000,1000000),
  2018:(-9000,1000000),
  2019:(-9000,1000000),
  2020:(-9000,1000000),
  2021:(-9000,1000000),
  2022:(-9000,1000000),
  2023:(-9000,1000000)}
Fisk_parameters = pd.DataFrame(index=years, columns=["alpha", "beta", "c"])
InvG_parameters = pd.DataFrame(index=years, columns=["alpha", "beta", "c"])
CS_InvG_parameters = pd.DataFrame({"alpha": [], "beta": [], "phi": []})
CSS_InvG_parameters = pd.DataFrame(index=years, columns=["alpha"])
CSS_bootstrap_naive = pd.DataFrame()
CSS_bootstrap_Jol = pd.DataFrame()
CSS_bootstrap_Jol_sim = pd.DataFrame()
CSS_bootstrap_se = pd.DataFrame(index=years)
def trim_data(data, var, year):
  if not isinstance(data, pd.DataFrame):
    raise TypeError("The data should be a DataFrame")
  if var not in data.columns:
    raise KeyError("Variable of interest is not a column in the data")
  bounds = dat_max_min[year]
  return data[(data[var] >= bounds[0]) &
              (data[var] <= bounds[1])].reset_index(drop=True).dropna()
###############
## Load Data ##
###############
print()
the_time()
if do_load_data:
  print("Loading data:")
  for i in years:
    print("Year {0}".format(i))
    temp_data = pd.read_csv("data_{0}.txt".format(i), header=0)  # read file
    temp_data = trim_data(temp_data, "income", i)                # trim ends
    temp_data = temp_data[temp_data["income"] != 0]              # remove 0s
    temp_data = temp_data[temp_data["weight"] >= 0]              # weight > 0
    data[i] = temp_data
    ecdfs[i] = est.make_ecdf(data[i], "income", "weight")
else:
  print("Skipping loading data")
print()
###############
## 2023 Data ##
###############
the_time()
if do_2023_short or do_2023_long:
  if 2023 in data:  # if loaded data_2023.txt
    data_2023 = data[2023]
  else:             # otherwise, load it now
    data_2023 = pd.read_csv("data_2023.txt", header=0)
    data_2023 = trim_data(data_2023, "income", 2023)
    data_2023 = data_2023[data_2023["income"] != 0]
    data_2023 = data_2023[data_2023["weight"] >= 0]
  print("Estimating parameters for 2023 data:")
  if do_2023_short:
    f = open("2023_parameters_short.txt", "w")
    for model in dist:
      if model != "CS_InvG" and model != "CSS_InvG" and \
         model != "LogN_P_mix" and model != "GB2":
        print(model)
        f.write("---{0}---\n".format(model))
        temp = est.estimator[model](data_2023, "income", "weight")
        for i in temp:
          f.write("{0}: {1}\n".format(i, temp[i]))
    f.close()
  else:
    print("Skipping Dagum, Burr, Fisk, InvG, Davis, and LogN_P_cut")
  if do_2023_long:
    f = open("2023_parameters_long.txt", "w")
    for model in ["GB2", "LogN_P_mix"]:
      print(model)
      f.write("---{0}---\n".format(model))
      temp = est.estimator[model](data_2023, "income", "weight")
      for i in temp:
        f.write("{0}: {1}\n".format(i, temp[i]))
    f.close()
  else:
    print("Skipping GB2 and LogN_P_mix")
else:
  print("Skipping estimating 2023 data")
print()
##########
## Fisk ##
##########
the_time()
if do_Fisk:
  if not do_load_data:
    raise RuntimeError("Please load data before estimating Fisk")
  print("Estimating Fisk parameters:")
  for i in years:
    print("Year {0}".format(i))
    p = est.estimator["Fisk"](data[i], "income", "weight",
      ecdf=ecdfs[i], x="income", y="weight")["parameters"]
    Fisk_parameters.loc[i] = p
  Fisk_parameters.to_csv("Fisk_parameters.txt")
else:
  print("Skipping Fisk estimation")
print()
###################
## Inverse Gamma ##
###################
the_time()
if do_InvG:
  if not do_load_data:
    raise RuntimeError("Please load data before estimating InvG")
  print("Estimating inverse-gamma parameters:")
  for i in years:
    print("Year {0}".format(i))
    p = est.estimator["InvG"](data[i], "income", "weight",
      ecdf=ecdfs[i], x="income", y="weight")["parameters"]
    InvG_parameters.loc[i] = p
  InvG_parameters.to_csv("InvG_parameters.txt")
else:
  print("Skipping inverse-gamma estimation")
print()
##################################
## Constant-shift Inverse Gamma ##
##################################
the_time()
if do_CS_InvG:
  if not do_load_data:
    raise RuntimeError("Please load data before estimating CS_InvG")
  print("Estimating constant-scale inverse-gamma parameters:")
  phi = (InvG_parameters["beta"] * InvG_parameters["c"]).sum() / \
    (InvG_parameters["beta"] ** 2).sum()
  for i in years:
    print("Year {0}".format(i))
    p = est.estimator["CS_InvG"](data[i], phi, "income", "weight",
      ecdf=ecdfs[i], x="income", y="weight")["parameters"]
    CS_InvG_parameters = pd.concat([CS_InvG_parameters,
      pd.DataFrame({"alpha": p[0], "beta": p[1], "phi": phi}, index=[i])])
  CS_InvG_parameters.to_csv("CS_InvG_parameters.txt")
else:
  print("Skipping constant-shift inverse-gamma estimation")
print()
########################################
## Constant-shift-scale Inverse Gamma ##
########################################
the_time()
if do_CSS_InvG:
  if not do_load_data:
    raise RuntimeError("Please load data before estimating CSS_InvG")
  print("Checking constants")
  InvG_parameters = pd.read_csv("InvG_parameters.txt", header=0, index_col=0)
  temp = cc.main(InvG_parameters.index, InvG_parameters,
    "alpha", "beta", "c")["linear"]
  phi = temp["phi"]
  psi = [temp["psi0"], temp["psi1"], temp["psi2"]]
  f = open("CSS_InvG_constants.txt", "w")
  for i in temp:
    f.write("{0},{1}\n".format(i, temp[i]))
  f.close()
  print("Estimating constant-shift-scale inverse-gamma parameters:")
  for i in years:
    print("Year {0}".format(i))
    p = est.estimator["CSS_InvG"](data[i], i, phi, psi,
      InvG_parameters.loc[i, "alpha"], "income", "weight", ecdf=ecdfs[i],
      x="income", y="weight")["parameters"]
    CSS_InvG_parameters.loc[i] = p
  CSS_InvG_parameters.to_csv("CSS_InvG_prop_parameters.txt")
else:
  print("Skipping constant-shift-scale inverse-gamma distribution")
print()
################
## bootstraps ##
################
the_time()
if do_bootstrap:
  if not do_load_data:
    raise RuntimeError("Please load data before bootstrapping")
  print("Bootstrapping standard errors:")
  InvG_parameters = pd.read_csv("InvG_parameters.txt", header=0, index_col=0)
  f = open("CSS_InvG_constants.txt")
  for line in f:
    temp = line[:-1].split(",")
    exec("{0} = {1}".format(temp[0], temp[1]))
  f.close()
  CSS_bootstrap_naive.index = range(100)
  CSS_bootstrap_Jol.index = range(100)
  CSS_bootstrap_Jol_sim.index = range(100)
  for i in years:
    CSS_pos_args = [est.estimator["CSS_InvG"], data[i], "income", "weight"]
    CSS_named_args = {"t": i, "phi": phi, "psi": [psi0, psi1],
      "a0": InvG_parameters.loc[i,"alpha"]}
    print("Year {0}".format(i))
    temp = pd.DataFrame(bootstrap.bootstrap_naive(
        *CSS_pos_args,
        **CSS_named_args)
      )["parameters"]
    temp = temp.explode().astype(np.float64)
    CSS_bootstrap_naive[i] = temp
    temp = pd.DataFrame(bootstrap.bootstrap_Jol(
        *CSS_pos_args,
        "region", "household",
        **CSS_named_args)
      )["parameters"]
    temp = temp.explode().astype(np.float64)
    CSS_bootstrap_Jol[i] = temp
    temp = pd.DataFrame(bootstrap.bootstrap_Jol_sim(
        *CSS_pos_args,
        "region",
        **CSS_named_args)
      )["parameters"]
    temp = temp.explode().astype(np.float64)
    CSS_bootstrap_Jol_sim[i] = temp
  for i in ["naive", "Jol", "Jol_sim"]:
    temp = eval("CSS_bootstrap_" + i)
    temp.to_csv("CSS_bootstrap_" + i + ".txt", index=False)
    iqrs = 0.7413 * (temp.quantile(0.75, numeric_only=True) -
                     temp.quantile(0.25, numeric_only=True))
    CSS_bootstrap_se[i] = iqrs
  CSS_bootstrap_se.to_csv("CSS_bootstrap_se.txt")
  print("se's are:")
  print(CSS_bootstrap_se)
else:
  print("Skipping bootstrapping")
print()
#############
## figures ##
#############
the_time()
if do_figures:
  if not do_load_data:
    raise RuntimeError("Please load data before making figures")
  print("Making figures")
  InvG_parameters = pd.read_csv("InvG_parameters.txt", header=0, index_col=0)
  CSS_InvG_lin_parameters = \
    pd.read_csv("CSS_InvG_lin_parameters.txt", header=0, index_col=0)
  CSS_InvG_prop_parameters = \
    pd.read_csv("CSS_InvG_prop_parameters.txt", header=0, index_col=0)
  Fisk_parameters = pd.read_csv("Fisk_parameters.txt", header=0, index_col=0)
  f = open("CSS_InvG_lin_constants.txt")
  for line in f:
    temp = line[:-1].split(",")
    exec("{0}_lin = {1}".format(temp[0], temp[1]))
  f.close()
  f = open("CSS_InvG_prop_constants.txt")
  for line in f:
    temp = line[:-1].split(",")
    exec("{0}_prop = {1}".format(temp[0], temp[1]))
  f.close()
  print("Making CSS_lin_densities.pdf")
  year1, year2, year3 = [1967, 1995, 2023]
  for i in [year1, year2, year3]:
    exec("""def F_{0}(x):
      return est.density["CSS_InvG"](x, {0}, phi_lin,
        [psi0_lin, psi1_lin, psi2_lin],
        CSS_InvG_lin_parameters.loc[{0}, "alpha"])""".format(i))
    exec("""c_{0} = psi0_lin + psi1_lin * {0} + \
      psi2_lin * CSS_InvG_lin_parameters.loc[{0}, "alpha"]""".format(i))
  make_figures.lin_loglog_graphs(
    data[year1], data[year2], data[year3], "income",
    [eval("F_{0}".format(year1))], [eval("c_{0}".format(year1))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year2))], [eval("c_{0}".format(year2))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year3))], [eval("c_{0}".format(year3))],
                                   [{"c": "black", "lw": 0.7}],
    [-20000,100000], [-0.05e-4,2.05e-4], [1000,1100000], [1e-10,2e-4],
    filename="CSS_lin_densities", wgt="weight",
    titles=["{0} Data (Linear Scale)".format(year1),
            "{0} Data (Log Scale)".format(year1),
            "{0} Data (Linear Scale)".format(year2),
            "{0} Data (Log Scale)".format(year2),
            "{0} Data (Linear Scale)".format(year3),
            "{0} Data (Log Scale)".format(year3)])
  print("Making CSS_prop_densities.pdf")
  year1, year2, year3 = [1967, 1995, 2023]
  for i in [year1, year2, year3]:
    exec("""def F_{0}(x):
      return est.density["CSS_InvG_prop"](x, {0}, phi_prop,
        [psi0_prop, psi1_prop],
        CSS_InvG_prop_parameters.loc[{0}, "alpha"])""".format(i))
    exec("""c_{0} = CSS_InvG_prop_parameters.loc[{0}, "alpha"] * \
      (psi0_prop + psi1_prop * {0})""".format(i))
  make_figures.lin_loglog_graphs(
    data[year1], data[year2], data[year3], "income",
    [eval("F_{0}".format(year1))], [eval("c_{0}".format(year1))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year2))], [eval("c_{0}".format(year2))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year3))], [eval("c_{0}".format(year3))],
                                   [{"c": "black", "lw": 0.7}],
    [-20000,100000], [-0.05e-4,2.05e-4], [1000,1100000], [1e-10,2e-4],
    filename="CSS_prop_densities", wgt="weight",
    titles=["{0} Data (Linear Scale)".format(year1),
            "{0} Data (Log Scale)".format(year1),
            "{0} Data (Linear Scale)".format(year2),
            "{0} Data (Log Scale)".format(year2),
            "{0} Data (Linear Scale)".format(year3),
            "{0} Data (Log Scale)".format(year3)])
  print("Making InvG_densities.pdf")
  year1, year2, year3 = [1967, 1995, 2023]
  for i in [year1, year2, year3]:
    exec("""def F_{0}(x):
      return est.density["InvG"](x, InvG_parameters.loc[{0}])""".format(i))
    exec("""c_{0} = InvG_parameters.loc[{0}, "c"]""".format(i))
  make_figures.lin_loglog_graphs(
    data[year1], data[year2], data[year3], "income",
    [eval("F_{0}".format(year1))], [eval("c_{0}".format(year1))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year2))], [eval("c_{0}".format(year2))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year3))], [eval("c_{0}".format(year3))],
                                   [{"c": "black", "lw": 0.7}],
    [-20000,100000], [-0.05e-4,2.05e-4], [1000,1100000], [1e-10,2e-4],
    filename="InvG_densities", wgt="weight",
    titles=["{0} Data (Linear Scale)".format(year1),
            "{0} Data (Log Scale)".format(year1),
            "{0} Data (Linear Scale)".format(year2),
            "{0} Data (Log Scale)".format(year2),
            "{0} Data (Linear Scale)".format(year3),
            "{0} Data (Log Scale)".format(year3)])
  print("Making Fisk_densities.pdf")
  year1, year2, year3 = [1967, 1995, 2023]
  for i in [year1, year2, year3]:
    exec("""def F_{0}(x):
      return est.density["Fisk"](x, Fisk_parameters.loc[{0}])""".format(i))
    exec("""c_{0} = Fisk_parameters.loc[{0}, "c"]""".format(i))
  make_figures.lin_loglog_graphs(
    data[year1], data[year2], data[year3], "income",
    [eval("F_{0}".format(year1))], [eval("c_{0}".format(year1))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year2))], [eval("c_{0}".format(year2))],
                                   [{"c": "black", "lw": 0.7}],
    [eval("F_{0}".format(year3))], [eval("c_{0}".format(year3))],
                                   [{"c": "black", "lw": 0.7}],
    [-20000,100000], [-0.05e-4,2.05e-4], [1000,1100000], [1e-10,2e-4],
    filename="Fisk_densities", wgt="weight",
    titles=["{0} Data (Linear Scale)".format(year1),
            "{0} Data (Log Scale)".format(year1),
            "{0} Data (Linear Scale)".format(year2),
            "{0} Data (Log Scale)".format(year2),
            "{0} Data (Linear Scale)".format(year3),
            "{0} Data (Log Scale)".format(year3)])
  print("Making InvG_parameter_graphs.pdf")
  norm_alpha = InvG_parameters["alpha"] / InvG_parameters["alpha"].sum()
  norm_beta = InvG_parameters["beta"] / InvG_parameters["beta"].sum()
  norm_c = InvG_parameters["c"] / InvG_parameters["c"].sum()
  make_figures.four_graphs(
    [years], [InvG_parameters["alpha"]], [plt.plot],
      [{"c": "black", "lw": 0.5}],
    [years], [InvG_parameters["beta"]], [plt.plot],
      [{"c": "black", "lw": 0.5}],
    [years], [InvG_parameters["c"]], [plt.plot],
      [{"c": "black", "lw": 0.5}],
    [*[years]*3],
      [norm_alpha, norm_beta, norm_c], [plt.plot, plt.plot, plt.plot],
      [{"c": "blue", "lw": 0.7, "ls": "--", "label": "Shape"},
       {"c": "black", "lw": 0.5, "label": "Scale"},
       {"c": "red", "lw": 1.0, "ls": ":", "label": "Shift"}],
    filename="InvG_parameter_graphs", titles=["Shape Parameter",
      "Scale Parameter", "Shift Parameter", "Normalized Parameters"],
      extra_code = \
"""plt.subplot(2,2,4)
plt.legend()""")
  print("Making InvG_parameter_regression.pdf")
  beta_hat = (psi0_lin + psi1_lin * InvG_parameters.index + \
    psi2_lin * InvG_parameters["alpha"]) / phi_lin
  c_hat = psi0_lin + psi1_lin * InvG_parameters.index + \
    psi2_lin * InvG_parameters["alpha"]
  norm_beta_hat = beta_hat / beta_hat.sum()
  norm_c_hat = c_hat / c_hat.sum()
  diff_beta_alpha = (norm_beta - norm_alpha).to_numpy()
  diff_c_alpha = (norm_c - norm_alpha).to_numpy()
  quot_beta_alpha = (norm_beta / norm_alpha).to_numpy()
  quot_c_alpha = (norm_c / norm_alpha).to_numpy()
  year_cons = pd.DataFrame({"cons": 1, "year": years}).to_numpy()
  reg_diff = np.linalg.inv(np.transpose(year_cons) @ year_cons) @ \
    np.transpose(year_cons) @ (0.5 * (diff_beta_alpha + diff_c_alpha))
  reg_diff_vals = reg_diff[0] + reg_diff[1] * InvG_parameters.index
  reg_quot = np.linalg.inv(np.transpose(year_cons) @ year_cons) @ \
    np.transpose(year_cons) @ (0.5 * (quot_beta_alpha + quot_c_alpha))
  reg_quot_vals = reg_quot[0] + reg_quot[1] * InvG_parameters.index
  from matplotlib.markers import MarkerStyle
  make_figures.four_graphs(
    [*[years]*3],
    [diff_beta_alpha, diff_c_alpha, reg_diff_vals],
      [plt.scatter, plt.scatter, plt.plot],
      [{"c": "blue", "s": 2.0, "label": "$\\bar\\beta_t-\\bar\\alpha_t$"},
       {"c": "red", "s": 10.0, "label": "$\\bar c_t-\\bar\\alpha_t$",
         "marker": "2"},
       {"c": "black", "lw": 0.7, "ls": "--", "label": "Trendline"}],
    [*[years]*3],
    [quot_beta_alpha, quot_c_alpha, reg_quot_vals],
      [plt.scatter, plt.scatter, plt.plot],
      [{"c": "blue", "s": 2.0, "label": "$\\bar\\beta_t/\\bar\\alpha_t$"},
       {"c": "red", "s": 10.0, "label": "$\\bar c_t/\\bar\\alpha_t$",
         "marker": "2"},
       {"c": "black", "lw": 0.7, "ls": "--", "label": "Trendline"}],
    [*[years]*2],
    [InvG_parameters["beta"], beta_hat], [plt.plot, plt.plot],
      [{"c": "black", "lw": 0.5, "label": "Observed"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Predicted"}],
    [*[years]*2],
    [InvG_parameters["c"], c_hat], [plt.plot, plt.plot],
      [{"c": "black", "lw": 0.5, "label": "Observed"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Predicted"}],
    filename="InvG_parameter_regression",
    titles=["(Normalized) Differences",
      "(Normalized) Quotients", "Predicted Scale",
      "Predicted Shift"],
    extra_code= \
"""for i in range(1, 5):
  plt.subplot(2, 2, i)
  plt.legend()""")
  print("Making Fisk_parameters_normalized.pdf")
  Fisk_alpha_norm = Fisk_parameters["alpha"] / Fisk_parameters["alpha"].sum()
  Fisk_beta_norm = Fisk_parameters["beta"] / Fisk_parameters["beta"].sum()
  Fisk_c_norm = Fisk_parameters["c"] / Fisk_parameters["c"].sum()
  make_figures.single_graph_ext(
    [*[years]*3],
      [Fisk_alpha_norm, Fisk_beta_norm, Fisk_c_norm],
      [plt.plot, plt.plot, plt.plot],
      [{"c": "blue", "lw": 0.7, "ls": "--", "label": "Shape"},
       {"c": "black", "lw": 0.5, "label": "Scale"},
       {"c": "red", "lw": 1.0, "ls": ":", "label": "Shift"}],
    filename="Fisk_parameters_normalized", title="Normalized Fisk Parameters",
      extra_code = "plt.legend()")
  print("Making CSS_InvG_parameters_graph.pdf")
  make_figures.single_graph(CSS_InvG_lin_parameters, "alpha",
    filename="CSS_InvG_parameters_graph", title="Parameter Estimates")
  print("comparison_linear_graphs.pdf")
  parameters = {}
  def add_from_parameter_file(filename, parameter_dict):
    f = open(filename)
    curr_model = ""
    for line in f:
      if line[0] == "-":
        curr_model = line[3:-4]
      elif line[0] == "f":
        pass
      elif line[0] == "p":
        temp = line[13:-2].split(", ")
        parameter_dict[curr_model] = list(map(float, temp))
    f.close()
    return parameter_dict
  parameters = add_from_parameter_file("2023_parameters_short.txt", parameters)
  parameters = add_from_parameter_file("2023_parameters_long.txt", parameters)
  for i in ["GB2", "Dagum", "Burr", "Davis", "LogN_P_cut",
            "LogN_P_mix"]:
    exec("""def F_{0}(x):
      return est.density['{0}'](x, parameters['{0}'])""".format(i))
    exec("c_{0} = parameters['{0}'][-1]".format(i))
  def F_CSS_InvG(x):
    return est.density["CSS_InvG"](x, 2023, phi_lin,
      [psi0_lin, psi1_lin, psi2_lin],
      CSS_InvG_lin_parameters.loc[2023, "alpha"])
  c_CSS_InvG = psi0_lin + psi1_lin * 2023 + \
    psi2_lin * CSS_InvG_lin_parameters.loc[2023, "alpha"]
  make_figures.lin_graphs(*[data[2023]]*6, "income",
    [F_CSS_InvG, F_GB2], [c_CSS_InvG, c_GB2],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Gen Beta II"}],
    [F_CSS_InvG, F_Dagum], [c_CSS_InvG, c_Dagum],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Dagum"}],
    [F_CSS_InvG, F_Burr], [c_CSS_InvG, c_Burr],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Burr"}],
    [F_CSS_InvG, F_Davis], [c_CSS_InvG, c_Davis],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Davis"}],
    [F_CSS_InvG, F_LogN_P_cut], [c_CSS_InvG, c_LogN_P_cut],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Cutoff"}],
    [F_CSS_InvG, F_LogN_P_mix], [c_CSS_InvG, c_LogN_P_mix],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Mixture"}],
    [-20000,100000], [-0.05e-5,2.05e-5],
    filename="comparison_linear_graphs", wgt="weight",
    titles=["Gen Beta II", "Dagum", "Burr", "Davis",
      "Log-Normal/Pareto Cutoff", "Log-Normal/Pareto Mix"])
  print("Making comparison_loglog_graphs.pdf")
  make_figures.loglog_graphs(*[data[2023]]*6, "income",
    [F_CSS_InvG, F_GB2],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Gen Beta II"}],
    [F_CSS_InvG, F_Dagum],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Dagum"}],
    [F_CSS_InvG, F_Burr],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Burr"}],
    [F_CSS_InvG, F_Davis],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Davis"}],
    [F_CSS_InvG, F_LogN_P_cut],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Cutoff"}],
    [F_CSS_InvG, F_LogN_P_mix],
      [{"c": "black", "lw": 0.5, "label": "Inverse-Gamma"},
       {"c": "blue", "lw": 0.7, "ls": "--", "label": "Mixture"}],
    [1000,1100000], [1e-10,2e-4],
    filename="comparison_loglog_graphs", wgt="weight",
    titles=["Gen Beta II", "Dagum", "Burr", "Davis",
      "Log-Normal/Pareto Cutoff", "Log-Normal/Pareto Mix"])
  print("Making gini_compare_graphs.pdf")
  gini = 1 / (1 + (CSS_InvG_lin_parameters["alpha"] - 1) * phi_lin) * \
    est.G(CSS_InvG_lin_parameters["alpha"] - 0.5) / (np.sqrt(np.pi) *
    est.G(CSS_InvG_lin_parameters["alpha"]))
  gini_unshift = \
    est.G(CSS_InvG_lin_parameters["alpha"] - 0.5) / (np.sqrt(np.pi) *
    est.G(CSS_InvG_lin_parameters["alpha"]))
  make_figures.single_graph(pd.DataFrame({"gini": gini}), "gini",
    filename="gini_graph")
  make_figures.single_graph(pd.DataFrame({"gini": gini_unshift}), "gini",
    filename="gini_unshift_graph")
  right_singularity = 1 - 1 / phi_lin
  alpha_vals = np.linspace(0.6, right_singularity - 0.2, 200)
  gini_vals_shift = 1 / (1 + (alpha_vals - 1) * phi_lin) * \
    est.G(alpha_vals - 0.5) / (np.sqrt(np.pi) *
    est.G(alpha_vals))
  gini_vals_unshift = \
    est.G(alpha_vals - 0.5) / (np.sqrt(np.pi) *
    est.G(alpha_vals))
  def D_gini(x):
    return est.psi(x - 0.5) - est.psi(x) - phi_lin / (1 + (x - 1) * phi_lin)
  min_gini = est.root(D_gini, bracket=[1, right_singularity - 0.2]).root
  min_alpha = CSS_InvG_lin_parameters["alpha"].min()
  max_alpha = CSS_InvG_lin_parameters["alpha"].max()
  make_figures.four_graphs(
    [years, (years[0], years[-1]), (years[0], years[-1])],
      [CSS_InvG_lin_parameters["alpha"],
        (min_alpha, min_alpha), (max_alpha, max_alpha)],
      [plt.plot, plt.plot, plt.plot],
      [{"c": "black", "lw": 0.5},
       {"c": "red", "lw": 0.7, "ls": "--"},
       {"c": "blue", "lw": 0.7, "ls": "--"}],
    [years], [gini], [plt.plot],
      [{"c": "black", "lw": 0.5}],
    [alpha_vals, (min_alpha, min_alpha), (max_alpha, max_alpha)],
      [gini_vals_shift, (0,2), (0,2)], [plt.plot, plt.plot, plt.plot],
      [{"c": "black", "lw": 0.5},
       {"c": "red", "lw": 0.7, "ls": "--"},
       {"c": "blue", "lw": 0.7, "ls": "--"}],
    [alpha_vals, (min_alpha, min_alpha), (max_alpha, max_alpha)],
      [gini_vals_unshift, (0,2), (0,2)], [plt.plot, plt.plot, plt.plot],
      [{"c": "black", "lw": 0.5},
       {"c": "red", "lw": 0.7, "ls": "--"},
       {"c": "blue", "lw": 0.7, "ls": "--"}],
    filename="gini_compare_graphs",
    titles=["Shape Parameter",
      "Gini Coefficient from Parameters",
      "Gini Coefficient Function (With Shift)",
      "Gini Coefficient Function (No Shift)"],
    extra_code= \
"""plt.subplot(2, 2, 3)
plt.ylim([0,2])
plt.gca().add_patch(mpl.patches.Rectangle([{min_gini},0],
  {max_alpha} - {min_gini}, 2, color="aliceblue"))
plt.subplot(2, 2, 4)
plt.ylim([0,2])""".format(min_gini=min_gini, max_alpha=alpha_vals[-1]))
else:
  print("Skipping figures")
print()
## Code for testing
if do_test:
  the_time()
  print("Testing...")

  # This section is intentionally blank

  print()
the_time()
print("End of main.py")
print()
## 
##
## End of file `main.py'.
