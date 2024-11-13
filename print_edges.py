##
## This is file `print_edges.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `edges')
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
for i in range(1967, 2024):
  if i != 1970:
    print("YEAR", i)
    d = pd.read_csv("data_{0}.txt".format(i), header=0)["income"]
    print(d.head(30))
    print(d.tail(30))
    print("\n\n\n")
## 
##
## End of file `print_edges.py'.
