##
## This is file `make_figures.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `figures')
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
import bin
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.rcParams["legend.fontsize"] = "small"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["text.usetex"] = True      # tells Pyplot to use TeX
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.format"] = "pdf"
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
def triple_graph(data, shape, scale, shift, *, filename, titles=[]):
  plt.close(plt.gcf())
  check_list(titles)
  if len(titles) > 0 and len(titles) < 3:
    raise ValueError("Please specify zero or all titles")
  elif len(titles) == 0:
    called_with_titles = False
  else:
    called_with_titles = True
  check_var(data, shape)
  check_var(data, scale)
  check_var(data, shift)
  plt.figure()
  grid = mpl.gridspec.GridSpec(4, 4, figure=plt.gcf())
  grid_boxes = [grid[0:2, 0:2], grid[0:2, 2:4], grid[2:4, 1:3]]
  parameter_names = [shape, scale, shift]
  for i in range(3):
    plt.subplot(grid_boxes[i])
    plt.plot(data.index, data[parameter_names[i]], lw=0.5, c="black")
    if called_with_titles:
      plt.title(titles[i])
  #plt.tight_layout()
  #plt.show()
  plt.gcf().set_size_inches(6.5, 5)
  plt.savefig(filename)
  plt.close(plt.gcf())
def lin_loglog_graphs(data1, data2, data3, var,
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
    d = eval("data" + str(i))
    check_var(d, var)
    binned_data = bin.bin_data(d, var, wgt)
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
    data4, data5, data6, var,
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
    d = eval("data" + str(i))
    check_var(d, var)
    binned_data = bin.bin_data(d, var, wgt)
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
    data4, data5, data6, var,
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
    d = eval("data" + str(i))
    check_var(d, var)
    binned_data = bin.bin_data(d, var, wgt)
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
## 
##
## End of file `make_figures.py'.
