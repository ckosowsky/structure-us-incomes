##
## This is file `bootstrap.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `bootstrap')
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
import numpy as np
import pandas as pd
def check_column(data, col):
  if not isinstance(data, pd.DataFrame):
    raise TypeError("Data should be a DataFrame.")
  if col not in data.columns:
    raise KeyError("{0} is not a column in the data.".format(col))
generator = np.random.default_rng(12345)
def bootstrap_naive(F, df, var, wgt=None, n=100, **kwargs):
  data = df.copy()
  check_column(data, var)
  #data.sort_values(var, inplace=True)
  if isinstance(wgt, type(None)):
    with_wgt = False
  else:
    check_column(data, wgt)
    with_wgt = True
  if with_wgt:
    data[wgt] = data[wgt] / data[wgt].sum() * len(data)
    lam = data[wgt]
  else:
    lam = 1
  new_weights = pd.DataFrame(
    generator.poisson(lam=lam, size=[n,len(data)]).transpose(),
    index=data.index,
    columns=["_brw" + str(i) for i in range(n)])
  data = pd.concat([data, new_weights], axis=1)
  temp = [F(data, var=var, wgt="_brw"+str(i), **kwargs) for i in range(n)]
  return temp
def bootstrap_Jol(F, df, var, wgt, strat, clust, n=100, **kwargs):
  data = df.copy()
  check_column(data, var)
  check_column(data, strat)
  check_column(data, clust)
  check_column(data, wgt)
  #data.sort_values(var, inplace=True)
  household_incomes = {}
  clusters_per_strat = {}
  strata = data[strat].unique()
  weight_per_strat = data.groupby(strat).sum()[wgt]
  data["_id"] = 0
  data["_weight_per_strat"] = 0
  data["_clusters_per_strat"] = 0
  for s in strata:
    household_incomes[s] = \
      data[data[strat] == s].groupby(clust).sum()[[var]].sort_values(var)
    household_incomes[s]["_id"] = np.arange(len(household_incomes[s])) // 4
    clusters_per_strat[s] = household_incomes[s]["_id"].max() + 1
    data.loc[data[strat] == s, "_weight_per_strat"] = weight_per_strat[s]
    data.loc[data[strat] == s, "_clusters_per_strat"] = clusters_per_strat[s]
    for h in household_incomes[s].index:
      data.loc[data[clust] == h, "_id"] = household_incomes[s].loc[h, "_id"]
  new_weights = pd.DataFrame(
    generator.poisson(lam=(data[wgt] * data["_clusters_per_strat"] /
      data["_weight_per_strat"]),
      size=[n,len(data)]).transpose().astype(float),
    index=data.index,
    columns=["_brw" + str(i) for i in range(n)])
  for s in strata:
    for c in range(clusters_per_strat[s]):
      new_weights.loc[(data[strat] == s) & (data["_id"] == c)] = \
        np.array(new_weights.loc[(data[strat] == s) &
                                 (data["_id"] == c)].sum(axis=0))
  for s in strata:
    new_weights[data[strat] == s] = (
      new_weights[data[strat] == s]
      * weight_per_strat[s]
      / np.array(new_weights[data[strat] == s].sum()))
  data = pd.concat([data, new_weights], axis=1)
  temp = [F(data, var=var, wgt="_brw"+str(i), **kwargs) for i in range(n)]
  return temp
def bootstrap_Jol_sim(F, df, var, wgt, strat, clust_size=10, n=100, **kwargs):
  data = df.copy()
  check_column(data, var)
  check_column(data, strat)
  check_column(data, wgt)
  #data.sort_values(var, inplace=True)
  clusters_per_strat = {}
  strata = data[strat].unique()
  weight_per_strat = data.groupby(strat).sum()[wgt]
  data["_id"] = 0
  data["_weight_per_strat"] = 0
  data["_clusters_per_strat"] = 0
  for s in strata:
    temp = data[data[strat] == s].sort_values(var)
    temp["_id"] = np.arange(len(temp)) // clust_size
    clusters_per_strat[s] = temp["_id"].max() + 1
    data.loc[data[strat] == s, "_id"] = temp["_id"]
    data.loc[data[strat] == s, "_weight_per_strat"] = weight_per_strat[s]
    data.loc[data[strat] == s, "_clusters_per_strat"] = clusters_per_strat[s]
  new_weights = pd.DataFrame(
    generator.poisson(lam=(data[wgt] * data["_clusters_per_strat"] /
      data["_weight_per_strat"]),
      size=[n,len(data)]).transpose().astype(float),
    index=data.index,
    columns=["_brw" + str(i) for i in range(n)])
  for s in strata:
    for c in range(clusters_per_strat[s]):
      new_weights.loc[(data[strat] == s) & (data["_id"] == c)] = \
        np.array(new_weights.loc[(data[strat] == s) &
                                 (data["_id"] == c)].sum(axis=0))
  for s in strata:
    new_weights[data[strat] == s] = (
      new_weights[data[strat] == s]
      * weight_per_strat[s]
      / np.array(new_weights[data[strat] == s].sum()))
  data = pd.concat([data, new_weights], axis=1)
  temp = [F(data, var=var, wgt="_brw"+str(i), **kwargs) for i in range(n)]
  return temp
## 
##
## End of file `bootstrap.py'.
