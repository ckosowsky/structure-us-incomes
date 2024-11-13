##
## This is file `bin.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `bin')
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
import pandas as pd
from numpy import exp
from numpy import log
def bin_data(data, var, wgt=None, *, cutoff=57000,
             num_linear_from_0=10, factor=1.2):
  if not isinstance(data, pd.DataFrame):
    raise ValueError("First argument of bin_data should be a DataFrame")
  if var not in data:
    raise KeyError("Variable to bin is not a column in the data")
  if isinstance(wgt, type(None)):
    called_with_wgt = False
  else:
    if wgt not in data:
      raise KeyError("Weight variable is not a column in the data")
    called_with_wgt = True
  data_to_use = data.sort_values(var).reset_index(drop=True)
  n = len(data_to_use)
  lin_width = cutoff / (num_linear_from_0 - 0.5)
  min = data_to_use.loc[(0, var)]
  max = data_to_use.loc[(n-1, var)]
  num_linear_bins = int((cutoff - min) // lin_width)
  remainder = (cutoff - min) % lin_width
  bins = [min]
  temp = min + remainder
  while temp < max and temp <= cutoff:
    bins.append(temp)
    temp = temp + lin_width
  temp = cutoff * factor
  while temp < max:
    bins.append(temp)
    temp = temp * factor
  next_bins = bins.copy()
  next_bins.pop(0)
  next_bins.append(max)
  binned = pd.DataFrame({"left":bins, "right":next_bins})
  binned["mid"] = 0.5 * (binned.left + binned.right)
  binned["width"] = binned.right - binned.left
  mass_vals = [0 for _ in bins]
  i = 0
  for j in range(len(bins)):
    k = 0
    while data_to_use.loc[(i, var)] < next_bins[j]:
      if called_with_wgt:  # if weight was specified, use survey weights
        k = k + data_to_use.loc[(i, wgt)]
      else:                # otherwise, each point adds mass of 1
        k = k + 1
      i = i+1
    mass_vals[j] = k
  k = 0
  i = n-1
  while data_to_use.loc[(i, var)] == max:
    if called_with_wgt:
      k = k + data_to_use.loc[(i, wgt)]
    else:
      k = k + 1
    i = i-1
  mass_vals[-1] = mass_vals[-1] + k
  binned["mass"] = mass_vals
  binned["freq"] = binned.mass / binned.mass.sum()
  binned["dens"] = binned.freq / binned.width
  return binned
## 
##
## End of file `bin.py'.
