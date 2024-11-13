##
## This is file `income_figs.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## income_figs.dtx  (with options: `py')
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
## This file originated from software that is designed
## to use a data product that was made using internal U.S.
## Census Bureau data. Any views expressed are those of
## the author and not those of the U.S. Census Bureau. The
## Census Bureau has reviewed this data product to ensure
## appropriate access, use, and disclosure avoidance
## protection of the confidential source data used to
## produce this product. This research was performed at a
## Federal Statistical Research Data Center under FSRDC
## Project Number 2679. (CBDRB-FY24-P2679-R11429)
## 
## 
## 
import pandas as pd
sample_dens = {}
for k,v in enumerate([1967, 1995, 2023]):
  temp_edges = pd.read_excel("output_request.xlsx",
    sheet_name="Sheet{0}_bin_edges_{1}".format(2*k + 1, v),
    skiprows=2, header=0)
  temp_freq = pd.read_excel("output_request.xlsx",
    sheet_name="Sheet{0}_bin_freq_{1}".format(2*k + 2, v),
    skiprows=2, header=0)
  temp_freq["freq"] = temp_freq["freq"] / (temp_edges["right"] -
    temp_edges["left"])
  sample_dens[v] = pd.DataFrame(
    {"mid": (temp_edges["right"] + temp_edges["left"]) / 2,
     "dens": temp_freq["freq"]})
CSS_InvG_lin_constants = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet7_CSS_InvG_lin_constants",
  skiprows=2, header=None, names=["val"], index_col=0)
phi_lin = CSS_InvG_lin_constants.at["phi", "val"]
psi0_lin = CSS_InvG_lin_constants.at["psi0", "val"]
psi1_lin = CSS_InvG_lin_constants.at["psi1", "val"]
psi2_lin = CSS_InvG_lin_constants.at["psi2", "val"]
CSS_InvG_lin_parameters = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet8_CSS_InvG_lin_parameters",
  skiprows=2, header=0, index_col=0)
parameters = {}
temp = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet10_estimates_2023",
  skiprows=2, header=None)
def pull_from_params(row, num):
  return [temp.at[row,1 + i] for i in range(num)]
parameters["Dagum"] = pull_from_params(2,4)
parameters["Burr"] = pull_from_params(5,4)
parameters["Fisk"] = pull_from_params(8, 3)
parameters["InvG"] = pull_from_params(11, 3)
parameters["Davis"] = pull_from_params(14, 3)
parameters["LogN_P_cut"] = pull_from_params(17, 6)
parameters["GB2"] = pull_from_params(20, 5)
parameters["LogN_P_mix"] = pull_from_params(23, 6)
Fisk_parameters = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet11_Fisk_parameters",
  skiprows=2, header=0, index_col=0)
InvG_parameters = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet12_InvG_parameters",
  skiprows=2, header=0, index_col=0)
CSS_InvG_prop_constants = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet13_CSS_InvG_prop_constants",
  skiprows=2, header=None, names=["val"], index_col=0)
phi_prop = CSS_InvG_prop_constants.at["phi", "val"]
psi0_prop = CSS_InvG_prop_constants.at["psi0", "val"]
psi1_prop = CSS_InvG_prop_constants.at["psi1", "val"]
CSS_InvG_prop_parameters = pd.read_excel("output_request.xlsx",
  sheet_name = "Sheet14_CSS_InvG_prop_parameter",
  skiprows=2, header=0, index_col=0)
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
pi    = np.pi
e     = np.e
exp   = np.exp
floor = np.floor
log   = np.log
sqrt  = np.sqrt
B     =  sp.beta
erf   =  sp.erf
G     =  sp.gamma
I     =  sp.betainc
Phi   =  sp.ndtr
Phinv =  sp.ndtri
psi   =  sp.digamma
Q     =  sp.gammaincc
root = opt.root_scalar
def psi1(x):
  return  sp.polygamma(1,x)
def Phi_prime(x):
  return exp(-x**2 / 2) / sqrt(2 * pi)
def zeta(x):
  return 1 + sp.zetac(x)
def zeta(x):
  return 1 +  sp.zetac(x)
Gamma_frac1 = 53/210
Gamma_frac2 = 24/7
def log_G(z):
  return log(G(z))
def log_G_approx(z):
  x = z - 1
  term1 = 0.5 * log(2 * pi * x)
  term2 = x * (log(x) - 1)
  term3 = (x * x + Gamma_frac1)
  term4 = log(1 + 1 / (12 * (x * x * x) + Gamma_frac2 * (x) - 0.5))
  return term1 + term2 + term3 * term4
density  = {}
def density_GB2(x, params):
  a, b, p, q, c = params
  if p < 10:
    log_G_p = log_G(p)
  else:
    log_G_p = log_G_approx(p)
  if q < 10:
    log_G_q = log_G(q)
  else:
    log_G_q = log_G_approx(q)
  if p + q < 10:
    log_G_pq = log_G(p + q)
  else:
    log_G_pq = log_G_approx(p + q)
  if x <= c:
    return 0
  else:
    num   = log(a) + (a*p-1) * log(x-c)           # numerator
    denom = (    (a*p) * log(b) + log_G_p
            + log_G_q - log_G_pq
            + (p+q) * log(1+((x-c)/b)**a)    )    # denominator
    return exp(num - denom)
def density_Dagum(x, params):
  a, b, p, c = params
  if x <= c:
    return 0
  else:
    num = log(a) + log(p) + (a*p-1) * log(x-c)
    denom = (a*p) * log(b) + (p+1) * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def density_Burr(x, params):
  a, b, q, c = params
  if x <= c:
    return 0
  else:
    num = log(a) + log(q) + (a-1) * log(x-c)
    denom = a * log(b) + (1+q) * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def density_Fisk(x, params):
  a, b, c = params
  if x <= c:
    return 0
  else:
    num   = log(a) + (a-1) * log(x-c)
    denom = a * log(b) + 2 * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def density_InvG(x, params):
  a, b, c = params
  if a < 10:
    log_G_a = log_G(a)
  else:
    log_G_a = log_G_approx(a)
  if x <= c:
    return 0
  else:
    num = a * log(b) - b / (x-c)
    denom = log_G_a + (1+a) * log(x-c)
    return exp(num - denom)
def density_Davis(x, params):
  a, b, c = params
  if x <= c + 1000:
    return 0
  else:
    num = a * log(b)
    denom = log(G(a)) + log(zeta(a)) + \
            log(exp(exp(log(b) - log(x - c))) - 1) + \
            (1 + a) * log(x - c)
    return exp(num - denom)
def density_CSS_InvG(x, t, phi, psi, a):
  psi0, psi1, psi2 = psi
  beta = (psi0 + psi1 * t + psi2 * a) / phi
  c = psi0 + psi1 * t + psi2 * a
  if x <= c:
    return 0
  else:
    return density_InvG(x, [a, beta, c])
def density_CSS_InvG_prop(x, t, phi, psi, a):
  psi0, psi1 = psi
  beta = a * (psi0 + psi1 * t) / phi
  c = a * (psi0 + psi1 * t)
  if x <= c:
    return 0
  else:
    return density_InvG(x, [a, beta, c])
def density_LogN_P_cut(x, params):
  mu, sigma_sq, k, x_m, a, c = params
  if k < c:
    print("k was less than c; setting k=c")
    k = c
  if x <= c:
    return 0
  elif x > c and x < k:
    frac = 1/((x-c) * sqrt(2 * pi * sigma_sq))
    exponent = -(log(x-c) - mu)**2 / (2 * sigma_sq)
    return frac * exp(exponent)
  elif x >= k:
    if x <= x_m + c:
      return 0
    else:
      return exp(log(a) + a * log(x_m) - (1+a) * log(x-c))
  else:
    raise RuntimeError("Something weird happened")
def density_LogN_P_mix(x, params):
  mu, sigma_sq, gamma, x_m, alpha, c = params
  if gamma < 0 or gamma > 1:
    raise ValueError("gamma is outside unit interval")
  if x <= c:
    return 0
  else:
    log_n_term = exp(log(gamma) - log(x - c) -
                     0.5 * log(2 * pi * sigma_sq) -
                     (log(x - c) - mu) ** 2 / (2 * sigma_sq))
    if x <= x_m + c:
      pareto_term = 0
    else:
      pareto_term = exp(log(1 - gamma) + log(alpha) + alpha * log(x_m) -
                        (1 + alpha) * log(x - c))
    return log_n_term + pareto_term
density["GB2"] = density_GB2
density["Dagum"] = density_Dagum
density["Burr"] = density_Burr
density["Fisk"] = density_Fisk
density["InvG"] = density_InvG
density["Davis"] = density_Davis
density["CSS_InvG"] = density_CSS_InvG
density["CSS_InvG_prop"] = density_CSS_InvG_prop
density["LogN_P_cut"] = density_LogN_P_cut
density["LogN_P_mix"] = density_LogN_P_mix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True      # tells Pyplot to use TeX
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.format"] = "pdf"
def check_var(data, var):
  if var not in data:
    raise KeyError("{0} is not a column in the data".format(var))
def check_list(x):
  is_list_like = hasattr(x, "__len__") and \
                 hasattr(x, "__getitem__") and \
                 hasattr(x, "__iter__")
  if not is_list_like or isinstance(x, str):
    raise TypeError("Please use list instead of {0} for {1}".format(type(x),x))
def single_graph(data, parameter, *, filename, title=None):
  plt.close(plt.gcf())
  if isinstance(title, type(None)):
    called_with_title = False
  else:
    called_with_title = True
  check_var(data, parameter)
  plt.plot(data.index, data[parameter], c="black", lw=0.5)
  if called_with_title:
    plt.title(title)
  #plt.show()
  plt.gcf().set_size_inches(3.25, 2.5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def lin_loglog_graphs(data1, data2, data3,
    list_F1, list_c1, list_opt1,
    list_F2, list_c2, list_opt2,
    list_F3, list_c3, list_opt3,
    x_bounds_lin, y_bounds_lin, x_bounds_log, y_bounds_log, *,
    filename, wgt=None, titles=[], with_legend=False):
  plt.close(plt.gcf())
  check_list(titles)
  if len(titles) > 0 and len(titles) < 6:
    raise ValueError("Please specify zero or all titles")
  elif len(titles) == 0:
    called_with_titles = False
  else:
    called_with_titles = True
  for i in range(1, 4):
    list_F = eval("list_F" + str(i))
    list_c = eval("list_c" + str(i))
    list_opt = eval("list_opt" + str(i))
    check_list(list_F)
    check_list(list_c)
    check_list(list_opt)
    binned_data = eval("data" + str(i))
    plt.subplot(3, 2, 2 * i - 1)
    if called_with_titles:
      plt.title(titles[2 * i - 2])
    for F, c, opt in zip(list_F, list_c, list_opt):
      x_vals = np.linspace(c, x_bounds_lin[1], 200)
      y_vals = list(map(F, x_vals))
      plt.plot(x_vals, y_vals, **opt)
    plt.scatter(binned_data["mid"], binned_data["dens"], s=2, c="blue")
    plt.xlim(x_bounds_lin)
    plt.ylim(y_bounds_lin)
    if with_legend:
      plt.legend(loc="upper right")
    plt.subplot(3, 2, 2 * i)
    if called_with_titles:
      plt.title(titles[2 * i - 1])
    x_vals = np.geomspace(*x_bounds_log, 200)
    for F, c, opt in zip(list_F, list_c, list_opt):
      y_vals = list(map(F, x_vals))
      plt.plot(x_vals, y_vals, **opt)
    plt.scatter(binned_data["mid"], binned_data["dens"], s=2, c="blue")
    plt.xlim(x_bounds_log)
    plt.ylim(y_bounds_log)
    plt.loglog()
    if with_legend:
      plt.legend(loc="lower left")
  #plt.show()
  plt.gcf().set_size_inches(6.5, 7.5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def lin_graphs(data1, data2, data3,
    data4, data5, data6,
    list_F1, list_c1, list_opt1,
    list_F2, list_c2, list_opt2,
    list_F3, list_c3, list_opt3,
    list_F4, list_c4, list_opt4,
    list_F5, list_c5, list_opt5,
    list_F6, list_c6, list_opt6,
    x_bounds, y_bounds, *,
    filename, wgt=None, titles=[], with_legend=True):
  plt.close(plt.gcf())
  check_list(titles)
  if len(titles) > 0 and len(titles) < 6:
    raise ValueError("Please specify zero or all titles")
  elif len(titles) == 0:
    called_with_titles = False
  else:
    called_with_titles = True
  for i in range(1, 7):
    list_F = eval("list_F" + str(i))
    list_c = eval("list_c" + str(i))
    list_opt = eval("list_opt" + str(i))
    check_list(list_F)
    check_list(list_c)
    check_list(list_opt)
    binned_data = eval("data" + str(i))
    plt.subplot(3, 2, i)
    if called_with_titles:
      plt.title(titles[i - 1])
    for F, c, opt in zip(list_F, list_c, list_opt):
      x_vals = np.linspace(c, x_bounds[1], 200)
      y_vals = list(map(F, x_vals))
      plt.plot(x_vals, y_vals, **opt)
    plt.scatter(binned_data["mid"], binned_data["dens"], s=2, c="blue")
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    if with_legend:
      plt.legend(loc="upper right")
  #plt.show()
  plt.gcf().set_size_inches(6.5, 7.5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def loglog_graphs(data1, data2, data3,
    data4, data5, data6,
    list_F1, list_opt1,
    list_F2, list_opt2,
    list_F3, list_opt3,
    list_F4, list_opt4,
    list_F5, list_opt5,
    list_F6, list_opt6,
    x_bounds, y_bounds, *,
    filename, wgt=None, titles=[], with_legend=True):
  plt.close(plt.gcf())
  check_list(titles)
  if len(titles) > 0 and len(titles) < 6:
    raise ValueError("Please specify zero or all titles")
  elif len(titles) == 0:
    called_with_titles = False
  else:
    called_with_titles = True
  for i in range(1, 7):
    list_F = eval("list_F" + str(i))
    list_opt = eval("list_opt" + str(i))
    check_list(list_F)
    check_list(list_opt)
    binned_data = eval("data" + str(i))
    plt.subplot(3, 2, i)
    if called_with_titles:
      plt.title(titles[i - 1])
    for F, opt in zip(list_F, list_opt):
      x_vals = np.geomspace(*x_bounds, 200)
      y_vals = list(map(F, x_vals))
      plt.plot(x_vals, y_vals, **opt)
    plt.scatter(binned_data["mid"], binned_data["dens"], s=2, c="blue")
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.loglog()
    if with_legend:
      plt.legend(loc="lower left")
  #plt.show()
  plt.gcf().set_size_inches(6.5, 7.5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def four_graphs(
    list_x1, list_y1, list_plot1, list_opt1,
    list_x2, list_y2, list_plot2, list_opt2,
    list_x3, list_y3, list_plot3, list_opt3,
    list_x4, list_y4, list_plot4, list_opt4, *,
    filename, titles=[], extra_code=""):
  plt.close(plt.gcf())
  check_list(titles)
  if len(titles) > 0 and len(titles) < 4:
    raise ValueError("Please specify zero or all titles")
  elif len(titles) == 0:
    called_with_titles = False
  else:
    called_with_titles = True
  for i in range(1,5):
    list_x = eval("list_x" + str(i))
    list_y = eval("list_y" + str(i))
    list_plot = eval("list_plot" + str(i))
    list_opt = eval("list_opt" + str(i))
    check_list(list_x)
    check_list(list_y)
    check_list(list_plot)
    check_list(list_opt)
    plt.subplot(2, 2, i)
    for x, y, plot, opt in zip(list_x, list_y, list_plot, list_opt):
      plot(x, y, **opt)
    if called_with_titles:
      plt.title(titles[i-1])
  exec(extra_code)
  #plt.show()
  plt.gcf().set_size_inches(6.5, 5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def single_graph_ext(
    list_x, list_y, list_plot, list_opt, *,
    filename, title=None, extra_code=""):
  plt.close(plt.gcf())
  if isinstance(title, type(None)):
    called_with_title = False
  else:
    called_with_title = True
  for x, y, plot, opt in zip(list_x, list_y, list_plot, list_opt):
    plot(x, y, **opt)
  if called_with_title:
    plt.title(title)
  exec(extra_code)
  #plt.show()
  plt.gcf().set_size_inches(3.25, 2.5)
  plt.savefig(filename)
  plt.close(plt.gcf())
print("Making CSS_lin_densities.pdf")
year1, year2, year3 = [1967, 1995, 2023]
for i in [year1, year2, year3]:
  exec("""def F_{0}(x):
    return density["CSS_InvG"](x, {0}, phi_lin,
      [psi0_lin, psi1_lin, psi2_lin],
      CSS_InvG_lin_parameters.loc[{0}, "alpha"])""".format(i))
  exec("""c_{0} = psi0_lin + psi1_lin * {0} + \
    psi2_lin * CSS_InvG_lin_parameters.loc[{0}, "alpha"]""".format(i))
lin_loglog_graphs(
  sample_dens[year1], sample_dens[year2], sample_dens[year3],
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
    return density["CSS_InvG_prop"](x, {0}, phi_prop,
      [psi0_prop, psi1_prop],
      CSS_InvG_prop_parameters.loc[{0}, "alpha"])""".format(i))
  exec("""c_{0} = CSS_InvG_prop_parameters.loc[{0}, "alpha"] * \
    (psi0_prop + psi1_prop * {0})""".format(i))
lin_loglog_graphs(
  sample_dens[year1], sample_dens[year2], sample_dens[year3],
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
    return density["InvG"](x, InvG_parameters.loc[{0}])""".format(i))
  exec("""c_{0} = InvG_parameters.loc[{0}, "c"]""".format(i))
lin_loglog_graphs(
  sample_dens[year1], sample_dens[year2], sample_dens[year3],
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
    return density["Fisk"](x, Fisk_parameters.loc[{0}])""".format(i))
  exec("""c_{0} = Fisk_parameters.loc[{0}, "c"]""".format(i))
lin_loglog_graphs(
  sample_dens[year1], sample_dens[year2], sample_dens[year3],
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
years = InvG_parameters.index
norm_alpha = InvG_parameters["alpha"] / InvG_parameters["alpha"].sum()
norm_beta = InvG_parameters["beta"] / InvG_parameters["beta"].sum()
norm_c = InvG_parameters["c"] / InvG_parameters["c"].sum()
four_graphs(
  [years], [InvG_parameters["alpha"]], [plt.plot],
    [{"c": "black", "lw": 0.5}],
  [years], [InvG_parameters["beta"]], [plt.plot],
    [{"c": "black", "lw": 0.5}],
  [years], [InvG_parameters["c"]], [plt.plot],
    [{"c": "black", "lw": 0.5}],
  [*[years]*3],
    [norm_alpha, norm_beta, norm_c], [plt.plot, plt.plot, plt.plot],
    [{"c": "blue", "lw": 0.7, "ls": "--", "label": "Shape $\\alpha$"},
     {"c": "black", "lw": 0.5, "label": "Scale $\\beta$"},
     {"c": "red", "lw": 1.0, "ls": ":", "label": "Shift $c$"}],
  filename="InvG_parameter_graphs", titles=["Shape Parameter $\\alpha$",
    "Scale Parameter $\\beta$", "Shift Parameter $c$",
    "Normalized Parameters"],
    extra_code = """plt.subplot(2,2,4)
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
four_graphs(
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
single_graph_ext(
  [*[years]*3],
    [Fisk_alpha_norm, Fisk_beta_norm, Fisk_c_norm],
    [plt.plot, plt.plot, plt.plot],
    [{"c": "blue", "lw": 0.7, "ls": "--", "label": "Shape $\\alpha$"},
     {"c": "black", "lw": 0.5, "label": "Scale $\\beta$"},
     {"c": "red", "lw": 1.0, "ls": ":", "label": "Shift $c$"}],
  filename="Fisk_parameters_normalized", title="Normalized Fisk Parameters",
    extra_code = "plt.legend()")
print("Making CSS_InvG_parameters_graph.pdf")
single_graph(CSS_InvG_lin_parameters, "alpha",
  filename="CSS_InvG_parameters_graph",
  title="Shape Parameter $\\alpha$ Estimates")
print("Making comparison_linear_graphs.pdf")
for i in ["GB2", "Dagum", "Burr", "Davis", "LogN_P_cut",
          "LogN_P_mix"]:
  exec("""def F_{0}(x):
    return density['{0}'](x, parameters['{0}'])""".format(i))
  exec("c_{0} = parameters['{0}'][-1]".format(i))
def F_CSS_InvG(x):
  return density["CSS_InvG"](x, 2023, phi_lin,
    [psi0_lin, psi1_lin, psi2_lin],
    CSS_InvG_lin_parameters.loc[2023, "alpha"])
c_CSS_InvG = psi0_lin + psi1_lin * 2023 + \
  psi2_lin * CSS_InvG_lin_parameters.loc[2023, "alpha"]
lin_graphs(*[sample_dens[2023]]*6,
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
loglog_graphs(*[sample_dens[2023]]*6,
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
  G(CSS_InvG_lin_parameters["alpha"] - 0.5) / (np.sqrt(np.pi) *
  G(CSS_InvG_lin_parameters["alpha"]))
gini_unshift = \
  G(CSS_InvG_lin_parameters["alpha"] - 0.5) / (np.sqrt(np.pi) *
  G(CSS_InvG_lin_parameters["alpha"]))
right_singularity = 1 - 1 / phi_lin
alpha_vals = np.linspace(0.6, right_singularity - 0.2, 200)
gini_vals_shift = 1 / (1 + (alpha_vals - 1) * phi_lin) * \
  G(alpha_vals - 0.5) / (np.sqrt(np.pi) *
  G(alpha_vals))
gini_vals_unshift = \
  G(alpha_vals - 0.5) / (np.sqrt(np.pi) *
  G(alpha_vals))
def D_gini(x):
  return psi(x - 0.5) - psi(x) - phi_lin / (1 + (x - 1) * phi_lin)
min_gini = root(D_gini, bracket=[1, right_singularity - 0.2]).root
min_alpha = CSS_InvG_lin_parameters["alpha"].min()
max_alpha = CSS_InvG_lin_parameters["alpha"].max()
four_graphs(
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
  titles=["Shape Parameter $\\alpha$",
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
## 
##
## End of file `income_figs.py'.
