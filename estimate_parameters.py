##
## This is file `estimate_parameters.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `estim')
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
import numpy          as np
import pandas         as pd
import scipy.optimize as opt
import scipy.special  as spec
import time
#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None
def the_time():
  print("The time is", time.asctime())
pi    = np.pi
e     = np.e
exp   = np.exp
floor = np.floor
log   = np.log
sqrt  = np.sqrt
min   = opt.minimize
min_s = opt.minimize_scalar
root  = opt.root_scalar
B     = spec.beta
erf   = spec.erf
G     = spec.gamma
I     = spec.betainc
Phi   = spec.ndtr
Phinv = spec.ndtri
psi   = spec.digamma
Q     = spec.gammaincc
def psi1(x):
  return spec.polygamma(1,x)
def Phi_prime(x):
  return exp(-x**2 / 2) / sqrt(2 * pi)
def zeta(x):
  return 1 + sp.zetac(x)
def zeta(x):
  return 1 + spec.zetac(x)
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
distribution = {}  # cumulative distribution functions
density      = {}  # densities
likelihood   = {}  # likelihoods
estimator    = {}  # estimators
def validate_var_wgt(data, var, wgt):
  if not isinstance(data, pd.DataFrame):
    raise ValueError("First argument of function should be a DataFrame")
  if var not in data:
    raise KeyError("Variable to analyze is not a column in the data")
  if wgt and wgt not in data:
    raise KeyError("Weight variable is not a column in the data")
def make_ecdf(data, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  temp = data.copy()
  if not wgt:
    wgt = "y"
    if var == "y":
      wgt = "y1"
    temp[wgt] = 1
  if not temp[var].is_monotonic_increasing:
    temp = temp.sort_values(var).reset_index(drop=True)
  temp = temp[[var, wgt]].groupby(var).sum()  # get unique var values
  temp.index.set_names(None, inplace=True)
  temp = temp.cumsum() / temp[wgt].sum()      # turn wgt into cdf
  temp[var] = temp.index                      # add var vals back into data
  temp = pd.concat([temp, temp]).sort_values(var).reset_index(drop=True)
  temp[wgt] = temp[wgt].rename(lambda x: x + 1)
  temp.loc[(0, wgt)] = 0
  return temp
def kolmogorov_smirnov(cdf, *, ecdf=None, x=None, y=None,
                               data=None, var=None, wgt=None):
  if isinstance(ecdf, type(None)) and isinstance(data, type(None)):
    raise ValueError("Please specify ecdf or data to use Kolmogorov-Smirnov")
  elif isinstance(ecdf, type(None)) and not isinstance(data, type(None)):
    validate_var_wgt(data, var, wgt)
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if not wgt and var != "y":
      y = "y"
    elif not wgt and var == "y":
      y = "y1"
    elif wgt:
      y = wgt
  elif not isinstance(ecdf, type(None)):
    validate_var_wgt(ecdf, x, None)
    validate_var_wgt(ecdf, y, None)
    ecdf = ecdf.copy()
  else:
    raise RuntimeError("Something weird happened")
  #ecdf = ecdf[(ecdf[y] > 0) & (ecdf[y] < 1)]
  vals = cdf(np.array(ecdf[x]))  # model cdf values
  emp_vals = np.array(ecdf[y])   # empitical cdf values
  #avg = (vals + ecdf[y]) / 2
  #weights = (1 / (avg * (1-avg))) ** (1/2)
  #weights = 1
  return np.max(abs(vals - emp_vals))
def fisk_moments(data, var, wgt=None):
  if wgt:
    n = data.weight.sum()
    m = (1/n) * (data[wgt] * log(data[var])).sum()
    v = (1/n) * (data[wgt] * (log(data[var]) - m)**2).sum()
  else:
    n = len(data)
    m = (1/n) * log(data[var]).sum()
    v = (1/n) * ((log(data[var]) - m)**2).sum()
  a = pi / sqrt(3 * v)
  b = exp(m)
  return [a, b]
def cdf_GB2(x, params):
  a, b, p, q, c = params
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(-a * (log(x[~mask] - c) - log(b)))
  x[~mask] = I(p, q, 1/(1+frac))
  return x
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
def L_GB2(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
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
  if wgt:
    n = data[wgt].sum()
    term1 = n * log(a)
    term2 = (a*p - 1) * (data[wgt] * log(data[var] - c)).sum()
    term3 = -n * a * p * log(b)
    term4 = -n * (log_G_p + log_G_q - log_G_pq)
    term5 = -(p+q) * (data[wgt] * log(1 + ((data[var] - c)/b)**a)).sum()
  else:
    n = len(data)
    term1 = n * log(a)
    term2 = (a*p - 1) * log(data[var] - c).sum()
    term3 = -n * a * p * log(b)
    term4 = -n * (log_G_p + log_G_q - log_G_pq)
    term5 = -(p+q) * log(1 + ((data[var] - c)/b)**a).sum()
  return term1 + term2 + term3 + term4 + term5
def L_GB2_unshift(data, params, var, wgt=None):
  a, b, p, q = params
  return L_GB2(data, [a, b, p, q, 0], var, wgt)
def estimate_GB2_find_pq(data, alpha, var, wgt):
  if wgt:
    n = data[wgt].sum()
    log_sum1 = (data[wgt] * log(1 + data[var] ** alpha)).sum() / n
    log_sum2 = alpha * (data[wgt] * log(data[var])).sum() / n
  else:
    n = len(data)
    log_sum1 = log(1 + data[var] ** alpha).sum() / n
    log_sum2 = alpha * log(data[var]).sum() / n
  def mle_foc(p_and_q):
    p, q = p_and_q
    p_term = psi(p + q) - psi(p) - log_sum1 + log_sum2
    q_term = psi(p + q) - psi(q) - log_sum1
    return [float(p_term), float(q_term)]
  def d_mle_foc(p_and_q):
    p, q = p_and_q
    return [[psi1(p + q) - psi1(p), psi1(p + q)          ],
            [psi1(p + q)          , psi1(p + q) - psi1(q)]]
  return opt.root(mle_foc, x0=[3, 1], jac=d_mle_foc).x
def estimate_GB2_find_a(data, var, wgt):
  def test_L(a):
    p, q = estimate_GB2_find_pq(data, a, var, wgt)
    return L_GB2(data, [a, 1, p, q, 0], var, wgt)
  alpha = min(lambda x: -test_L(x), x0=1, method="Nelder-Mead",
    bounds=[(0.1,None)]).x[0]
  p, q = estimate_GB2_find_pq(data, alpha, var, wgt)
  return [alpha, p, q]
def estimate_GB2_fit_for_c(data, c, b, var, wgt, ecdf, x, y):
  stand_data = data.copy()
  stand_data[var] = (stand_data[var] - c) / b
  stand_data = stand_data[stand_data.income > 0]
  #print("finding a, p, q:", time.asctime())
  a, p, q = estimate_GB2_find_a(stand_data, var, wgt)
  #print("fitting model:", time.asctime())
  def F(x):
    return cdf_GB2(x, [a, b, p, q, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, p, q, c]}
  #print("done:", time.asctime())
  return temp
def estimate_GB2_get_cb(dict):
  return [dict["parameters"][-1], dict["parameters"][1]]
def estimate_GB2(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  b_vals = [10000 + 500*i for i in range(81)]
  c_vals = [-15000 + 500*i for i in range(21)]
  best_fits = []
  the_time()
  for b in b_vals:
    for c in c_vals:
      temp = estimate_GB2_fit_for_c(data, c, b, var, wgt, ecdf, x, y)
      if len(best_fits) < 10 and temp["parameters"][0] > 0.1:
        best_fits.append(temp)
      else:
        if temp["fit"] < best_fits[-1]["fit"] and temp["parameters"][0] > 0.1:
          i = 0
          while temp["fit"] > best_fits[i]["fit"]:
            i = i + 1
          best_fits.insert(i, temp)
          best_fits.pop()
  pairs = [estimate_GB2_get_cb(i) for i in best_fits]
  pairs = [(p[0] - 500 + 100*i, p[1]) for p in pairs for i in range(11)]
  pairs = [(p[0], p[1] - 500 + 100*i) for p in pairs for i in range(11)]
  pairs = set(pairs)
  best_fits = []
  the_time()
  for p in pairs:
    temp = estimate_GB2_fit_for_c(data, *p, var, wgt, ecdf, x, y)
    if len(best_fits) < 10 and temp["parameters"][0] > 0.1:
      best_fits.append(temp)
    elif temp["parameters"][0] > 0.1:
      if temp["fit"] < best_fits[-1]["fit"] and temp["parameters"][0] > 0.1:
        i = 0
        while temp["fit"] > best_fits[i]["fit"]:
          i = i + 1
        best_fits.insert(i, temp)
        best_fits.pop()
  pairs = [estimate_GB2_get_cb(i) for i in best_fits]
  pairs = [(p[0] - 100 + 10*i, p[1]) for p in pairs for i in range(21)]
  pairs = [(p[0], p[1] - 100 + 10*i) for p in pairs for i in range(21)]
  pairs = set(pairs)
  solution = {"fit":2}
  the_time()
  for p in pairs:
    temp = estimate_GB2_fit_for_c(data, *p, var, wgt, ecdf, x, y)
    if temp["fit"] < solution["fit"] and temp["parameters"][0] > 0.1:
      solution = temp
  the_time()
  return solution
distribution["GB2"] = cdf_GB2
density["GB2"]      = density_GB2
likelihood["GB2"]   = L_GB2
estimator["GB2"]    = estimate_GB2
def cdf_Dagum(x, params):
  a, b, p, c = params
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(-a * (log(x[~mask] - c) - log(b)))
  x[~mask] = (1 + frac) ** (-p)
  return x
def density_Dagum(x, params):
  a, b, p, c = params
  if x <= c:
    return 0
  else:
    num = log(a) + log(p) + (a*p-1) * log(x-c)
    denom = (a*p) * log(b) + (p+1) * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def L_Dagum(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b, p, c = params
  if wgt:
    n = data[wgt].sum()
    term1 = n * log(a)
    term2 = n * log(p)
    term3 = (a*p-1) * (data[wgt] * log(data[var] - c)).sum()
    term4 = -n * a * p * log(b)
    term5 = -(p+1) * (data[wgt] * log(1 + ((data[var] - c)/b)**a)).sum()
  else:
    n = len(data)
    term1 = n * log(a)
    term2 = n * log(p)
    term3 = (a*p-1) * log(data[var] - c).sum()
    term4 = -n * a * p * log(b)
    term5 = -(p+1) * log(1 + ((data[var] - c)/b)**a).sum()
  return term1 + term2 + term3 + term4 + term5
def L_Dagum_unshift(data, params, var, wgt=None):
  a, b, p = params
  return L_Dagum(data, [a, b, p, 0], var, wgt)
def estimate_Dagum_initial(data, var, wgt):
  a, b = fisk_moments(data, var, wgt)
  return [a, b, 1]
def estimate_Dagum_unshift(data, var, wgt):
  guess = estimate_Dagum_initial(data, var, wgt)
  def neg_L(params):
    return -L_Dagum_unshift(data, params, var, wgt)
  sol = min(neg_L, x0=guess, method="Nelder-Mead")
  a, b, p = sol.x
  return [a, b, p]
def estimate_Dagum_fit_for_c(data, c, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  a, b, p = estimate_Dagum_unshift(shift_data, var, wgt)
  def F(x):
    return cdf_Dagum(x, [a, b, p, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, p, c]}
  return temp
def estimate_Dagum(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c(c):
    return estimate_Dagum_fit_for_c(data, c, var, wgt, ecdf, x, y)["fit"]
  sol = min_s(check_c, bracket=[-20000,-5000], options={"xtol":1.4e-5})
  return estimate_Dagum_fit_for_c(data, sol.x, var, wgt, ecdf, x, y)
distribution["Dagum"] = cdf_Dagum
density["Dagum"]      = density_Dagum
likelihood["Dagum"]   = L_Dagum
estimator["Dagum"]    = estimate_Dagum
def cdf_Burr(x, params):
  a, b, q, c = params
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(a * (log(x[~mask] - c) - log(b)))
  x[~mask] = 1 - (1 + frac) ** (-q)
  return x
def density_Burr(x, params):
  a, b, q, c = params
  if x <= c:
    return 0
  else:
    num = log(a) + log(q) + (a-1) * log(x-c)
    denom = a * log(b) + (1+q) * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def L_Burr(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b, q, c = params
  if wgt:
    n = data[wgt].sum()
    term1 = n * log(a)
    term2 = n * log(q)
    term3 = (a-1) * (data[wgt] * log(data[var] - c)).sum()
    term4 = -n * a * log(b)
    term5 = -(q+1) * (data[wgt] * log(1 + ((data[var] - c)/b)**a)).sum()
  else:
    n = len(data)
    term1 = n * log(a)
    term2 = n * log(q)
    term3 = (a-1) * log(data[var] - c).sum()
    term4 = -n * a * log(b)
    term5 = -(q+1) * log(1 + ((data[var] - c)/b)**a).sum()
  return term1 + term2 + term3 + term4 + term5
def L_Burr_unshift(data, params, var, wgt=None):
  a, b, q = params
  return L_Burr(data, [a, b, q, 0], var, wgt)
def estimate_Burr_initial(data, var, wgt):
  a, b = fisk_moments(data, var, wgt)
  return [a, b, 1]
def estimate_Burr_unshift(data, var, wgt):
  guess = estimate_Burr_initial(data, var, wgt)
  def neg_L(params):
    return -L_Burr_unshift(data, params, var, wgt)
  sol = min(neg_L, x0=guess, method="Nelder-Mead")
  return sol.x
def estimate_Burr_fit_for_c(data, c, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  a, b, q = estimate_Burr_unshift(shift_data, var, wgt)
  def F(x):
    return cdf_Burr(x, [a, b, q, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, q, c]}
  return temp
def estimate_Burr(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c(c):
    return estimate_Burr_fit_for_c(data, c, var, wgt, ecdf, x, y)["fit"]
  sol = min_s(check_c, bracket=[-20000,-5000], options={"xtol":1.4e-5})
  return estimate_Burr_fit_for_c(data, sol.x, var, wgt, ecdf, x, y)
distribution["Burr"] = cdf_Burr
density["Burr"]      = density_Burr
likelihood["Burr"]   = L_Burr
estimator["Burr"]    = estimate_Burr
def cdf_Fisk(x, params):
  a, b, c = params
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(-a * (log(x[~mask] - c) - log(b)))
  x[~mask] = 1 / (1 + frac)
  return x
def density_Fisk(x, params):
  a, b, c = params
  if x <= c:
    return 0
  else:
    num   = log(a) + (a-1) * log(x-c)
    denom = a * log(b) + 2 * log(1 + ((x-c)/b)**a)
    return exp(num - denom)
def L_Fisk(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b, c = params
  if wgt:
    n = data[wgt].sum()
    term1 = n * log(a)
    term2 = (a-1) * (data[wgt] * log(data[var] - c)).sum()
    term3 = -n * a * log(b)
    term4 = -2 * (data[wgt] * log(1 + ((data[var]-c)/b)**a)).sum()
  else:
    n = len(data)
    term1 = n * log(a)
    term2 = (a-1) * log(data[var] - c).sum()
    term3 = -n * a * log(b)
    term4 = -2 * log(1 + ((data[var]-c)/b)**a).sum()
  return term1 + term2 + term3 + term4
def L_Fisk_unshift(data, params, var, wgt=None):
  a, b = params
  return L_Fisk(data, [a, b, 0], var, wgt)
estimate_Fisk_initial = fisk_moments
def estimate_Fisk_unshift(data, var, wgt):
  guess = estimate_Fisk_initial(data, var, wgt)
  def neg_L(params):
    return -L_Fisk_unshift(data, params, var, wgt)
  sol = min(neg_L, x0=guess, method="Nelder-Mead")
  return sol.x
def estimate_Fisk_fit_for_c(data, c, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  a, b = estimate_Fisk_unshift(shift_data, var, wgt)
  def F(x):
    return cdf_Fisk(x, [a, b, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, c]}
  return temp
def estimate_Fisk(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c(c):
    return estimate_Fisk_fit_for_c(data, c, var, wgt, ecdf, x, y)["fit"]
  sol = min_s(check_c, bracket=[data[var].min() - 2000, data[var].min()],
    options={"xtol":1.4e-5})
  return estimate_Fisk_fit_for_c(data, sol.x, var, wgt, ecdf, x, y)
distribution["Fisk"] = cdf_Fisk
density["Fisk"]      = density_Fisk
likelihood["Fisk"]   = L_Fisk
estimator["Fisk"]    = estimate_Fisk
def cdf_InvG(x, params):
  a, b, c = params
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(log(b) - log(x[~mask] - c))
  x[~mask] = Q(a, frac)
  return x
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
def L_InvG(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b, c = params
  if a < 10:
    log_G_a = log_G(a)
  else:
    log_G_a = log_G_approx(a)
  if wgt:
    n = data[wgt].sum()
    term1 = n * a * log(b)
    term2 = -n * log_G_a
    term3 = -b * (data[wgt] / (data[var] - c)).sum()
    term4 = -(1+a) * (data[wgt] * log(data[var] - c)).sum()
  else:
    n = len(data)
    term1 = n * a * log(b)
    term2 = -n * log_G_a
    term3 = -b * (1 / (data.income - c)).sum()
    term4 = -(1+a) * log(data.income - c).sum()
  return term1 + term2 + term3 + term4
def L_InvG_unshift(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b = params
  return L_InvG(data, [a, b, 0], var, wgt)
def estimate_InvG_unshift(data, var, wgt):
  if wgt:
    n = data[wgt].sum()
    recip_data = (1/n) * (data[wgt] / data[var]).sum()
    log_data = (1/n) * (data[wgt] * log(data[var])).sum()
  else:
    n = len(data)
    recip_data = (1/n) * (1 / data[var]).sum()
    log_data = (1/n) * log(data[var]).sum()
  def estimate_InvG_numeric(x):
    return log(x) - psi(x) - log(recip_data) - log_data
  temp = estimate_InvG_numeric(10)
  if temp < 0:
    temp1 = estimate_InvG_numeric(0.01)
    a = root(estimate_InvG_numeric, bracket=[0.01, 10],
             fprime = lambda x: (1/x) - psi1(x)).root
  elif temp == 0:
    a = 10
  else:
    prev_bound = 10
    curr_bound = 30
    temp = estimate_InvG_numeric(curr_bound)
    while temp > 0:
      prev_bound = curr_bound
      curr_bound = curr_bound + 20
      temp = estimate_InvG_numeric(curr_bound)
    if temp == 0:
      a = curr_bound
    else:
      a = root(estimate_InvG_numeric, bracket=[prev_bound,curr_bound],
               fprime = lambda x: (1/x) - psi1(x)).root
  b = a / recip_data
  return [a, b]
def estimate_InvG_fit_for_c(data, c, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  a, b = estimate_InvG_unshift(shift_data, var, wgt)
  def F(x):
    return cdf_InvG(x, [a, b, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, c]}
  return temp
def estimate_InvG(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c(c):
    return estimate_InvG_fit_for_c(data, c, var, wgt, ecdf, x, y)["fit"]
  sol = min_s(check_c, method="bounded", bounds=[-23000,-1000])
  return estimate_InvG_fit_for_c(data, sol.x, var, wgt, ecdf, x, y)
distribution["InvG"] = cdf_InvG
density["InvG"]      = density_InvG
likelihood["InvG"]   = L_InvG
estimator["InvG"]    = estimate_InvG
def cdf_Davis_sum(x, alpha):
  k = 1
  temp = Q(alpha, x)
  var = temp
  while np.max(np.abs(temp)) > 0.000001:
    k = k + 1
    temp = exp(-alpha * log(k) + Q(alpha, exp(log(k) + log(x))))
    var = var + temp
  return var
def cdf_Davis(x, params):
  a, b, c = params
  x = x.astype(float)
  mask = (x <= c)
  x[mask] = 0
  x[~mask] = cdf_Davis_sum(b / (np.array(x[~mask]) - c), a)
  return x / zeta(a)
def density_Davis(x, params):
  a, b, c = params
  if x <= c:
    return 0
  else:
    num = a * log(b)
    denom = log(G(a)) + log(zeta(a)) + \
            log(exp(exp(log(b) - log(x - c))) - 1) + \
            (1 + a) * log(x - c)
    return exp(num - denom)
def L_Davis(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b, c = params
  if isinstance(wgt, type(None)):  # if no weight provided
    n = len(data)
    term4 = -log(exp(exp(log(b) - log(data[var] - c))) - 1).sum()
    term5 = -(1 + a) * log(data[var] - c).sum()
  else:                            # if weight provided
    n = data[wgt].sum()
    term4 = -(data[wgt] *
      log(exp(exp(log(b) - log(data[var] - c))) - 1)).sum()
    term5 = -(1 + a) * (data[wgt] * log(data[var] - c)).sum()
  term1 =  n * a * log(b)
  term2 = -n * log(G(a))
  term3 = -n * log(zeta(a))
  return term1 + term2 + term3 + term4 + term5
def L_Davis_unshift(data, params, var, wgt=None):
  return L_Davis(data, [*params, 0], var, wgt)
def zeta_prime(z):
  k = 2
  temp = 1
  val = -log(k) / k ** z
  while np.abs(temp) > 0.000001:
    k = k + 1
    temp = -exp(log(log(k)) - z * log(k))
    val = val + temp
  return val
def estimate_Davis_a_from_b(data, b, var, wgt=None):
  if isinstance(wgt, type(None)):
    n = len(data)
    wgt = pd.Series(1, index=data.index)
  else:
    n = data[wgt].sum()
    wgt = data[wgt]
  temp = log(b) - (1/n) * (wgt * log(data[var])).sum()
  def dL_da(a):
    return temp - psi(a) - zeta_prime(a) / zeta(a)
  lower_bound = 2
  if dL_da(lower_bound) == 0:         # if lower_bound is a root
    return bound1
  elif dL_da(lower_bound) > 0:        # if lower_bound is too small
    upper_bound = 1.2 * lower_bound
    while dL_da(upper_bound) > 0:
      lower_bound = upper_bound
      upper_bound = 1.2 * upper_bound
  elif dL_da(lower_bound) < 0:        # if lower_bound is too big
    while dL_da(lower_bound) < 0:
      upper_bound = lower_bound
      lower_bound = lower_bound ** 0.8
  else:
    raise RuntimeError("Something weird happened")
  return opt.root_scalar(dL_da, x0=2,
    bracket=[lower_bound,upper_bound]).root
def estimate_Davis_unshift(data, var, wgt):
  guess = estimate_InvG_unshift(data, var, wgt)
  def neg_L_with_a_in_terms_of_b(b):
    a = estimate_Davis_a_from_b(data, b, var, wgt)
    return -L_Davis_unshift(data, [a,b], var, wgt)
  sol = opt.minimize_scalar(neg_L_with_a_in_terms_of_b,
    bracket=[guess[1] - 20000, guess[1]])
  return [estimate_Davis_a_from_b(data, sol.x, var, wgt), sol.x]
def estimate_Davis_fit_for_c(data, c, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  a, b = estimate_Davis_unshift(shift_data, var, wgt)
  def F(x):
    return cdf_Davis(x, [a, b, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [a, b, c]}
  return temp
def estimate_Davis(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c(c):
    return estimate_Davis_fit_for_c(data, c, var, wgt, ecdf, x, y)["fit"]
  sol = opt.minimize_scalar(check_c,
    bracket=[-20000,-7000], options={"xtol":1.4e-5})
  return estimate_Davis_fit_for_c(data, sol.x, var, wgt, ecdf, x, y)
distribution["Davis"] = cdf_Davis
density["Davis"]      = density_Davis
likelihood["Davis"]   = L_Davis
estimator["Davis"]    = estimate_Davis
def cdf_CS_InvG(x, phi, params):
  a, b = params
  shift = phi * b
  x = x.astype(np.float64)
  mask = x <= shift
  x[mask] = 0
  frac = exp(log(b) - log(x[~mask] - shift))
  x[~mask] = Q(a, frac)
  return x
def density_CS_InvG(x, phi, params):
  a, b = params
  shift = phi * b
  if x <= shift:
    return 0
  else:
    num = a * log(b) - b / (x - shift)
    denom = log_G(a) + (1 + alpha) * log(x - shift)
    return exp(num - denom)
def L_CS_InvG(data, phi, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  a, b = params
  shift = phi * b
  if wgt:
    n = data[wgt].sum()
    term1 = n * a * log(b)
    term2 = -n * log_G(a)
    term3 = -b * (data[wgt] / (data[var] - shift)).sum()
    term4 = -(1+a) * (data[wgt] * log(data[var] - shift)).sum()
  else:
    n = len(data)
    term1 = n * a * log(b)
    term2 = -n * log_G(a)
    term3 = -b * (1 / (data[var] - shift)).sum()
    term4 = -(1+a) * log(data[var] - shift).sum()
  return term1 + term2 + term3 + term4
def estimate_CS_InvG_alpha1(data, phi, beta, var, wgt):
  shift = phi * beta
  shift_data = data.copy()
  shift_data = shift_data[shift_data[var] > shift]
  shift_data[var] = shift_data[var] - shift
  if wgt:
    n = data[wgt].sum()
    sum = (shift_data[wgt] * log(beta / shift_data[var])).sum()
  else:
    n = len(data)
    sum = log(beta / shift_data[var]).sum()
  return (1/n) * sum
def estimate_CS_InvG_alpha2(data, phi, beta, var, wgt):
  shift = phi * beta
  shift_data = data.copy()
  shift_data = shift_data[shift_data[var] > shift]
  shift_data[var] = shift_data[var] - shift
  if wgt:
    n = data[wgt].sum()
    num1 = (1-phi) / n * (shift_data[wgt] * (shift_data[var] + shift) /
      shift_data[var] ** 2).sum()
    num2 = (phi**2) * (beta/n) * (shift_data[wgt] /
      shift_data[var] ** 2).sum()
    denom = (1/beta) + (phi/n) * (shift_data[wgt] / shift_data[var]).sum()
  else:
    n = len(data)
    num1 = (1-phi) / n * ((shift_data[var] + shift) /
      shift_data[var] ** 2).sum()
    num2 = (phi**2) * (beta/n) * (1 / shift_data[var] ** 2).sum()
    denom = (1/beta) + (phi/n) * (1 / shift_data[var]).sum()
  return (num1 + num2) / denom
def estimate_CS_InvG(data, phi, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def temp_numeric(b):
    return estimate_CS_InvG_alpha1(data, phi, b, var, wgt) - \
      psi(estimate_CS_InvG_alpha2(data, phi, b, var, wgt))
  right_bound = 200000
  left_bound = 100000
  temp = temp_numeric(left_bound)
  if temp > 0:
    rtemp = temp_numeric(right_bound)
    while rtemp > 0:
      left_bound = right_bound
      right_bound = right_bound * 1.05
      rtemp = temp_numeric(right_bound)
    if rtemp == 0:
      beta = right_bound
    else:
      beta = root(temp_numeric, bracket=[left_bound,right_bound]).root
  elif temp == 0:
    beta = left_bound
  else:
    while temp < 0:
      right_bound = left_bound
      left_bound = left_bound * 0.98
      temp = temp_numeric(left_bound)
    if temp == 0:
      beta = left_bound
    else:
      beta = root(temp_numeric, bracket=[left_bound,right_bound]).root
  alpha = estimate_CS_InvG_alpha2(data, phi, beta, var, wgt)
  def F(x):
    return cdf_CS_InvG(x, phi, [alpha, beta])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "phi": phi,
    "parameters": [alpha, beta]}
  return temp
distribution["CS_InvG"] = cdf_CS_InvG
density["CS_InvG"]      = density_CS_InvG
likelihood["CS_InvG"]   = L_CS_InvG
estimator["CS_InvG"]    = estimate_CS_InvG
def cdf_CSS_InvG(x, t, phi, psi, a):
  psi0, psi1, psi2 = psi
  b = (psi0 + psi1 * t + psi2 * a) / phi
  c = psi0 + psi1 * t + psi2 * a
  x = x.astype(np.float64)
  mask = x <= c
  x[mask] = 0
  frac = exp(log(b) - log(x[~mask] - c))
  x[~mask] = Q(a, frac)
  return x
def density_CSS_InvG(x, t, phi, psi, a):
  psi0, psi1, psi2 = psi
  beta = (psi0 + psi1 * t + psi2 * a) / phi
  c = psi0 + psi1 * t + psi2 * a
  if x <= c:
    return 0
  else:
    return density_InvG(x, [a, beta, c])
def L_CSS_InvG(data, t, phi, psi, a, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  psi0, psi1, psi2 = psi
  beta = (psi0 + psi1 * t + psi2 * a) / phi
  c = psi0 + psi1 * t + psi2 * a
  return L_InvG(data, [a, beta, c], var, wgt)
def estimate_CSS_InvG_fit_for_a(t, phi, psi, a, ecdf, x, y):
  psi0, psi1, psi2 = psi
  beta = (psi0 + psi1 * t + psi2 * a) / phi
  c = psi0 + psi1 * t + psi2 * a
  def F(x):
    return cdf_InvG(x, [a, beta, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "phi": phi,
    "psi": psi,
    "parameters": [a]}
  return temp
def estimate_CSS_InvG(data, t, phi, psi, a0, var, wgt=None, *, \
                      ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_a(a):
    return estimate_CSS_InvG_fit_for_a(t, phi, psi, a, ecdf, x, y)["fit"]
  sol = min(check_a, x0=a0).x[0]
  return estimate_CSS_InvG_fit_for_a(t, phi, psi, sol, ecdf, x, y)
distribution["CSS_InvG"] = cdf_CSS_InvG
density["CSS_InvG"]      = density_CSS_InvG
likelihood["CSS_InvG"]   = L_CSS_InvG
estimator["CSS_InvG"]    = estimate_CSS_InvG
def cdf_CSS_InvG_prop(x, t, phi, psi, a):
  psi0, psi1 = psi
  b = a * (psi0 + psi1 * t) / phi
  c = a * (psi0 + psi1 * t)
  x = x.astype(np.float64)
  mask = (x <= c)
  x[mask] = 0
  frac = exp(log(b) - log(x[~mask] - c))
  x[~mask] = Q(a, frac)
  return x
def density_CSS_InvG_prop(x, t, phi, psi, a):
  psi0, psi1 = psi
  beta = a * (psi0 + psi1 * t) / phi
  c = a * (psi0 + psi1 * t)
  if x <= c:
    return 0
  else:
    return density_InvG(x, [a, beta, c])
def L_CSS_InvG_prop(data, t, phi, psi, a, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  psi0, psi1 = psi
  beta = a * (psi0 + psi1 * t) / phi
  c = a * (psi0 + psi1 * t)
  return L_InvG(data, [a, beta, c], var, wgt)
def estimate_CSS_InvG_prop_fit_for_a(t, phi, psi, a, ecdf, x, y):
  if a <= 0:
    return {"fit": 1, "parameters": a}
  else:
    psi0, psi1 = psi
    beta = a * (psi0 + psi1 * t) / phi
    c = a * (psi0 + psi1 * t)
    def F(x):
      return cdf_InvG(x, [a, beta, c])
    temp = {
      "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
      "parameters": [a]}
    return temp
def estimate_CSS_InvG_prop(data, t, phi, psi, a0, var, wgt=None, *, \
                      ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_a(a):
    return estimate_CSS_InvG_prop_fit_for_a(t, phi, psi, a, ecdf, x, y)["fit"]
  sol = min(check_a, x0=a0).x[0]
  return estimate_CSS_InvG_prop_fit_for_a(t, phi, psi, sol, ecdf, x, y)
distribution["CSS_InvG_prop"] = cdf_CSS_InvG_prop
density["CSS_InvG_prop"]      = density_CSS_InvG_prop
likelihood["CSS_InvG_prop"]   = L_CSS_InvG_prop
estimator["CSS_InvG_prop"]    = estimate_CSS_InvG_prop
def cdf_LogN_P_cut(x, params):
  mu, sigma_sq, k, x_m, a, c = params
  x = x.astype(np.float64)
  mask_c = x <= c
  mask_k = x <= k
  mask_xm = x < x_m
  if k < c:
    raise RuntimeError("k was less than c; problem?")
  x[mask_c] = 0
  x[(~mask_c) & mask_k] = Phi(
    (log(x[(~mask_c) & mask_k] - c) - mu) / sqrt(sigma_sq))
  x[(~mask_k) & mask_xm] = 0
  x[(~mask_k) & (~mask_xm)] = \
    1 - (x_m / (x[(~mask_k) & (~mask_xm)] - c)) ** a
  return x
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
def L_LogN_P_cut(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  mu, sigma_sq, k, x_m, a, c = params
  data_down = data[data[var] < k]
  data_up = data[data[var] >= k]
  if wgt:
    n_down = data_below_k[wgt].sum()
    n_up = data_above_k[wgt].sum()
    down_terms = -(data_down[wgt] * log(data_down[var] - c)).sum() - \
      (n_down/2) * log(2 * sigma_sq * pi) - \
      (data_down[wgt] * (log(data_down[var] - c) - mu)**2).sum() * \
      (1 / (2*sigma_sq))
    up_terms = n_up * (log(a) + a * log(x_m)) + \
      (1+a) * (data_up[wgt] * log(data_up[var] - c)).sum()
  else:
    n_down = len(data_below_k)
    n_up = len(data_above_k)
    down_terms = -log(data_down[var] - c).sum() - \
      (n_down/2) * log(2 * sigma_sq * pi) - \
      ((log(data_down.income() - c) - mu)**2).sum() / (2*sigma_sq)
    up_terms = n_up * (log(a) + a * log(x_m)) + \
      (1+a) * log(data_up.income - c).sum()
  return down_terms + up_terms
def L_LogN_P_cut_unshift(data, params, var, wgt=None):
  mu, sigma_sq, k, x_m, a = params
  return L_LogN_P_cut(data, [mu, sigma_sq, k, x_m, a, 0], var, wgt)
def estimate_LogN_P_cut_cons(data, k, var, wgt):
  data_down = data[data[var] < k]
  data_up = data[data[var] >= k]
  if wgt:
    n_down = data_down[wgt].sum()
    n_up = data_up[wgt].sum()
    p2 = (1/n_down) * (data_down[wgt] * log(data_down[var])).sum()
    p0 = (1/n_down) * (data_down[wgt] * log(data_down[var])**2).sum() - \
      log(k) * p2
    p1 = log(k) - p2
  else:
    n_down = len(data_down)
    n_up = len(data_up)
    p2 = (1/n_down) * log(data_down[var]).sum()
    p0 = (1/n_down) * (log(data_down[var])**2).sum() - log(k) * p2
    p1 = log(k) - p2
  return [p0, p1, p2, n_down, n_up, k]
def estimate_LogN_P_cut_get_mu(cons):
  p0, p1, p2, n_down, n_up, k = cons
  def eq_to_solve(mu):
    ratio = (log(k) - mu) / sqrt(p0 + p1 * mu)
    if Phi(ratio) == 1:
      hazard = np.inf
    else:
      hazard = Phi_prime(ratio) / (1 - Phi(ratio))
    return n_down * p2 - n_down * mu + n_up * sqrt(p0 + p1 * mu) * hazard
  min_val = -p0 / p1 + 1.0e-6
  max_val = min_val
  while eq_to_solve(max_val) > 0:
    max_val = max_val + 2
  sol = root(eq_to_solve, bracket=[min_val, max_val])
  return sol.root
def estimate_LogN_P_cut_get_alpha(data, cons, var, wgt):
  p0, p1, p2, n_down, n_up, k = cons
  data_up = data[data[var] >= k]
  if wgt:
    temp = (data_up[wgt] * log(data_up[var] / k)).sum()
  else:
    temp = log(data_up[var] / k).sum()
  return n_up / temp
def estimate_LogN_P_cut_get_params(mu, alpha, cons):
  p0, p1, p2, n_down, n_up, k = cons
  sigma_sq = p0 + p1 * mu
  x_m = k * (1 - Phi((log(k) - mu) / sqrt(sigma_sq))) ** (1/alpha)
  return [sigma_sq, x_m]
def estimate_LogN_P_cut_unshift(data, k, var, wgt):
  constants     = estimate_LogN_P_cut_cons(data, k, var, wgt)
  mu            = estimate_LogN_P_cut_get_mu(constants)
  alpha         = estimate_LogN_P_cut_get_alpha(data, constants, var, wgt)
  sigma_sq, x_m = estimate_LogN_P_cut_get_params(mu, alpha, constants)
  return [mu, sigma_sq, x_m, alpha]
def estimate_LogN_P_cut_fit_for_c(data, c, k, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  shift_k = k - c
  mu, sigma_sq, x_m, alpha = estimate_LogN_P_cut_unshift(shift_data,
    shift_k, var, wgt)
  def F(x):
    return cdf_LogN_P_cut(x, [mu, sigma_sq, k, x_m, alpha, c])
  temp = {
    "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
    "parameters": [mu, sigma_sq, k, x_m, alpha, c]}
  return temp
def estimate_LogN_P_cut(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  def check_c_k(p):
    c, k = p
    f = estimate_LogN_P_cut_fit_for_c(data, c, k, var, wgt, ecdf, x, y)["fit"]
    return f
  sol = min(check_c_k, method="Nelder-Mead", x0=[-9000,110000],
            options={"xatol": 0.1})
  c, k = sol.x
  return estimate_LogN_P_cut_fit_for_c(data, c, k, var, wgt, ecdf, x, y)
distribution["LogN_P_cut"] = cdf_LogN_P_cut
density["LogN_P_cut"]      = density_LogN_P_cut
likelihood["LogN_P_cut"]   = L_LogN_P_cut
estimator["LogN_P_cut"]    = estimate_LogN_P_cut
def cdf_LogN_P_mix(x, params):
  mu, sigma_sq, gamma, x_m, alpha, c = params
  if gamma < 0 or gamma > 1:
    raise ValueError("gamma is outside unit interval")
  mask_c = x <= c
  mask_c_xm = x < x_m + c
  term1 = x.astype(np.float64)
  term1[mask_c] = 0
  term1[~mask_c] = Phi((log(term1[~mask_c] - c) - mu) / sqrt(sigma_sq))
  term2 = x.astype(np.float64)
  term2[mask_c_xm] = 0
  term2[~mask_c_xm] = \
    1 - exp(alpha * (log(x_m) - log(term2[~mask_c_xm] - c)))
  return gamma * term1 + (1 - gamma) * term2
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
def L_LogN_P_mix(data, params, var, wgt=None):
  validate_var_wgt(data, var, wgt)
  mu, sigma_sq, gamma, x_m, alpha, c = params
  if gamma < 0 or gamma > 1:
    raise ValueError("gamma is outside unit interval")
  data_down = data[data[var] <= x_m + c]
  data_up = data[data[var] > x_m + c]
  if wgt:
    down_terms = (data_down[wgt] *
      (log(gamma) - log(data_down[var]- c) - 0.5 * log(2 * pi * sigma_sq) -
      (log(data_down[var] - c) - mu) ** 2 / (2 * sigma_sq))).sum()
    up_terms = (data_up[wgt] * log(
      exp(log(gamma) - log(data_up[var] - c) -
          0.5 * log(2 * pi * sigma_sq) -
          (log(data_up[var] - c) - mu)**2 / (2 * sigma_sq)) +
      exp(log(1 - gamma) + log(alpha) + alpha * log(x_m) -
          (1 + alpha) * log(data_up[var] - c)))).sum()
  else:
    down_terms = (log(gamma) -
      log(data_down[var] - c) - 0.5 * log(2 * pi * sigma_sq) -
      (log(data_down[var] - c) - mu)**2 / (2 * sigma_sq)).sum()
    up_terms = log(
      exp(log(gamma) - log(data_up[var] - c) -
          0.5 * log(2 * pi * sigma_sq) -
          (log(data_up[var] - c) - mu)**2 / (2 * sigma_sq)) +
      exp(log(1 - gamma) + log(alpha) + alpha * log(x_m) -
          (1 + alpha) * log(data_up[var] - c))).sum()
  return down_terms + up_terms
def L_LogN_P_mix_unshift(data, params, wgt, var=None):
  mu, sigma_sq, gamma, x_m, alpha = params
  return L_LogN_P_mix(data, [mu, sigma_sq, gamma, x_m, alpha, 0], var, wgt)
def estimate_LogN_P_mix_logn_trunc(data, gamma, k, var, wgt):
  data_down = data[data[var] <= k]
  if wgt:
    n_down = data_down[wgt].sum()
    eta = n_down / data[wgt].sum()
    sum_log = (data_down[wgt] * log(data_down[var])).sum() / n_down
    sum_log_sq = (data_down[wgt] * log(data_down[var]) ** 2).sum() / n_down
  else:
    n_down = len(data_down)
    eta = n_down / len(data)
    sum_log = log(data_down[var]).sum() / n_down
    sum_log_sq = (log(data_down[var]) ** 2).sum() / n_down
  phi_coef = Phinv(eta / gamma)  # eta is empirical mass
  phi_coef_sq = phi_coef ** 2
  b = phi_coef_sq * (sum_log - log(k)) - 2 * log(k)
  c = phi_coef_sq * (log(k) * sum_log - sum_log_sq) + log(k) ** 2
  mu = (-b - sqrt(b ** 2 - 4 * c)) / 2
  sigma = (log(k) - mu) / phi_coef
  return [mu, sigma ** 2]
def estimate_LogN_P_mix_inner_params(data, gamma, x_m, c, var, wgt):
  data_above_c = data[data[var] > c]
  shift_data = data_above_c.copy()
  shift_data[var] = shift_data[var] - c
  mu, sigma_sq = estimate_LogN_P_mix_logn_trunc(shift_data,
    gamma, x_m, var, wgt)
  def neg_L(a):
    return -L_LogN_P_mix(data_above_c,
      [mu, sigma_sq, gamma, x_m, a, c], var, wgt)
  sol = min(neg_L, x0=1, bounds=[(0.5, 5)], method="Nelder-Mead")
  return [mu, sigma_sq, sol.x[0]]
def estimate_LogN_P_mix_find_gamma(data, c, x_m, var, wgt):
  data_up = data[data[var] > x_m + c]
  if wgt:
    n_down = data[data[var] <= x_m + c][wgt].sum()
    n_up = data_up[wgt].sum()
    eta = n_down / data[wgt].sum()
  else:
    n_down = len(data[data[var] <= x_m + c])
    n_up = len(data_up)
    eta = n_down / len(data)
  def dL(g):
    if g <= eta or g >= 1:
      print("bad g for dL")
      return float("nan")
    mu, sigma_sq, alpha = estimate_LogN_P_mix_inner_params(data,
      g, x_m, c, var, wgt)
    l = exp(log(alpha) + alpha * log(x_m) -
      alpha * log(data_up[var] - c) + 0.5 * log(2 * pi * sigma_sq) +
      (log(data_up[var] - c) - mu) ** 2 / (2 * sigma_sq))
    term1 = n_down / g
    term2 = (data_up.weight * (1 - l) / (g + (1 - g) * l)).sum()
    return {"val": term1 + term2, "parameters": [mu, sigma_sq, g, alpha]}
  max_g = min_s(lambda x: -dL(x)["val"], method="bounded",
    bounds=[0.99 * eta + 0.01, 0.01 * eta + 0.99]).x
  max_dL = dL(max_g)["val"]
  if max_dL == 0:
    return dL(max_g)["parameters"]
  elif max_dL > 0:
    delta = 0.01
    while dL((delta) * max_g + (1 - delta))["val"] > 0:
      delta = delta / 10
    bound = (delta) * max_g + (1 - delta)
    g = root(lambda x: dL(x)["val"], bracket=[max_g, bound], xtol=1e-5).root
    return dL(g)["parameters"]
  elif max_dL < 0:
    return False
def estimate_LogN_P_mix_fit_for_c(data, c, x_m, var, wgt, ecdf, x, y):
  shift_data = data.copy()
  shift_data[var] = shift_data[var] - c
  shift_data = shift_data[shift_data[var] > 0]
  temp_params = estimate_LogN_P_mix_find_gamma(data, c, x_m, var, wgt)
  if temp_params:
    mu, sigma_sq, gamma, alpha = temp_params
    def F(x):
      return cdf_LogN_P_mix(x, [mu, sigma_sq, gamma, x_m, alpha, c])
    temp = {
      "fit": kolmogorov_smirnov(F, ecdf=ecdf, x=x, y=y),
      "parameters": [mu, sigma_sq, gamma, x_m, alpha, c]}
    return temp
  else:
    print("bad gamma, xm = {0}, c = {1}".format(x_m, c))
    return {"fit": 1, "parameters": []}
def estimate_LogN_P_mix_get_cxm(dict):
  return [dict["parameters"][-1], dict["parameters"][3]]
def estimate_LogN_P_mix(data, var, wgt=None, *, ecdf=None, x=None, y=None):
  validate_var_wgt(data, var, wgt)
  if isinstance(ecdf, type(None)):
    ecdf = make_ecdf(data, var, wgt)
    x = var
    if wgt:
      y = wgt
    else:
      if var == "y":
        y = "y1"
      else:
        y = "y"
  x_m_vals = [35000 + 500*i for i in range(41)]  # xm in [35000, 55000]
  c_vals = [-12000 + 500*i for i in range(21)]   # c  in [-12000,-2000]
  best_fits = []
  the_time()
  print("First iteration of brute-force search")
  for x_m in x_m_vals:
    for c in c_vals:
      temp = estimate_LogN_P_mix_fit_for_c(data, c, x_m, var, wgt, ecdf, x, y)
      if len(best_fits) < 10:
        best_fits.append(temp)
      else:
        if temp["fit"] < best_fits[-1]["fit"]:
          i = 0
          while temp["fit"] > best_fits[i]["fit"]:
            i = i + 1
          best_fits.insert(i, temp)
          best_fits.pop()
  pairs = [estimate_LogN_P_mix_get_cxm(i) for i in best_fits]
  pairs = [(p[0] - 500 + 100*i, p[1]) for p in pairs for i in range(11)]
  pairs = [(p[0], p[1] - 500 + 100*i) for p in pairs for i in range(11)]
  pairs = set(pairs)
  best_fits = []
  the_time()
  print("Second iteration of brute-force search")
  for p in pairs:
    temp = estimate_LogN_P_mix_fit_for_c(data, *p, var, wgt, ecdf, x, y)
    if len(best_fits) < 10:
      best_fits.append(temp)
    else:
      if temp["fit"] < best_fits[-1]["fit"]:
        i = 0
        while temp["fit"] > best_fits[i]["fit"]:
          i = i + 1
        best_fits.insert(i, temp)
        best_fits.pop()
  pairs = [estimate_LogN_P_mix_get_cxm(i) for i in best_fits]
  pairs = [(p[0] - 100 + 10*i, p[1]) for p in pairs for i in range(21)]
  pairs = [(p[0], p[1] - 100 + 10*i) for p in pairs for i in range(21)]
  pairs = set(pairs)
  solution = {"fit":2}
  the_time()
  print("Third iteration of brute-force search")
  for p in pairs:
    temp = estimate_LogN_P_mix_fit_for_c(data, *p, var, wgt, ecdf, x, y)
    if temp["fit"] < solution["fit"]:
      solution = temp
  return solution
distribution["LogN_P_mix"] = cdf_LogN_P_mix
density["LogN_P_mix"]      = density_LogN_P_mix
likelihood["LogN_P_mix"]   = L_LogN_P_mix
estimator["LogN_P_mix"]    = estimate_LogN_P_mix
## 
##
## End of file `estimate_parameters.py'.
