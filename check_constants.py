##
## This is file `check_constants.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `constants')
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
import numpy as np
import numpy.linalg as la
def check_df(x):
  if not isinstance(x, pd.DataFrame):
    msg = """
The second arguments of main() should be a Pandas
DataFrame. Right now one it is {0}
instead.\n""".format(type(x))
    raise TypeError(msg)
def check_col(x, col):
  if col not in x.columns:
    msg = """
The third, fourth, and fifth arguments of main() should
be columns in data (second argument). However, it looks
like {0} is not a column in data.\n""".format(col)
    raise KeyError(msg)
def main(years, data, a_col, b_col, c_col):
  check_df(data)
  check_col(data, a_col)
  check_col(data, b_col)
  check_col(data, c_col)
  pd.options.mode.chained_assignment = None
  alpha = data[a_col].to_numpy()
  temp = data[a_col].to_numpy() * years.to_numpy()
  A_lin  = np.concatenate([np.ones([len(years), 1]),
    np.transpose([years]), np.transpose([alpha])], axis=1)
  A_prop = np.concatenate([np.transpose([alpha]),
    np.transpose([temp])], axis=1)
  B = data[b_col].to_numpy()
  C = data[c_col].to_numpy()
  phi = np.sum(C) / np.sum(B)
  psi_lin = 0.5 * la.inv(A_lin.transpose() @ A_lin) @ \
    A_lin.transpose() @ (phi * B + C)
  psi_prop = 0.5 * la.inv(A_prop.transpose() @ A_prop) @ \
    A_prop.transpose() @ (phi * B + C)
  return {
    "linear":
      {"phi": phi, "psi0": psi_lin[0],
       "psi1": psi_lin[1], "psi2": psi_lin[2]},
    "proportional":
      {"phi": phi, "psi0": psi_prop[0], "psi1": psi_prop[1]}}
## 
##
## End of file `check_constants.py'.
