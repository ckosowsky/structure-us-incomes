##
## This is file `gen_files.py',
## generated with the docstrip utility.
##
## The original source files were:
##
## estimate_parameters.dtx  (with options: `files')
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
def try_int(x):
  try:
    return int(x)
  except:
    return float("nan")
files = {
  #1964:"cpsmar64.dat",
  #1966:"cpsmar66.dat",
  1967:"cpsmar67.dat",
  1968:"cpsmar68.dat",
  1969:"cpsmar69.dat",
  1971:"cpsmar71.dat",
  1972:"cpsmar72.dat",
  1973:"cpsmar73.dat",
  1974:"cpsmar74.dat",
  1975:"cpsmar75.dat",
  1976:"cpsmar76.dat",
  1977:"cpsmar77.dat",
  1978:"cpsmar78.dat",
  1979:"cpsmar79.dat",
  1980:"cpsmar80.dat",
  1981:"cpsmar81.dat",
  1982:"cpsmar82.dat",
  1983:"cpsmar83.dat",
  1984:"cpsmar84.dat",
  1985:"cpsmar85.dat",
  1986:"cpsmar86.dat",
  1987:"cpsmar87.dat",
  1988:"cpsmar88.dat",
  1989:"cpsmar89.dat",
  1990:"cpsmar90.dat",
  1991:"cpsmar91.dat",
  1992:"cpsmar92.dat",
  1993:"cpsmar93.dat",
  1994:"cpsmar94.dat",
  1995:"cpsmar95.dat",
  1996:"cpsmar96.dat",
  1997:"cpsmar97.dat",
  1998:"mar98pub.cps",
  1999:"mar99pub.cps",
  2000:"mar00supp.dat",
  2001:"mar01supp.dat",
  2002:"mar02supp.dat",
  2003:"asec2003.pub",
  2004:"asec2004.pub",
  2005:"asec2005_pubuse.pub",
  2006:"asec2006_pubuse.pub",
  2007:"asec2007_pubuse_tax2.dat",
  2008:"asec2008_pubuse.dat",
  2009:"asec2009_pubuse.dat",
  2010:"asec2010_pubuse.dat",
  2011:"asec2011_pubuse.dat",
  2012:"asec2012_pubuse.dat",
  2013:"asec2013_pubuse.dat",
  2014:"asec2014_pubuse_3x8_rerun.dat",
  2015:"asec2015_pubuse.dat",
  2016:"asec2016_pubuse_v3.dat",
  2017:"asec2017_pubuse.dat",
  2018:"asec2018_pubuse.dat",
  2019:["hhpub19.csv","pppub19.csv"],
  2020:["hhpub20.csv","pppub20.csv"],
  2021:["hhpub21.csv","pppub21.csv"],
  2022:["hhpub22.csv","pppub22.csv"],
  2023:["hhpub23.csv","pppub23.csv"]}
regions = {
  1:3,   # Alabama
  2:4,   # Alaska
  # 3 is American Samoa
  4:4,   # Arizona
  5:3,   # Arkansas
  6:4,   # California
  # 7 is Canal Zone
  8:4,   # Colorado
  9:1,   # Connecticut
  10:3,  # Delaware
  11:3,  # DC
  12:3,  # Florida
  13:3,  # Georgia
  # 14 is Guam
  15:4,  # Hawaii
  16:4,  # Idaho
  17:2,  # Illinois
  18:2,  # Indiana
  19:2,  # Iowa
  20:2,  # Kansas
  21:3,  # Kentucky
  22:3,  # Louisiana
  23:1,  # Maine
  24:3,  # Maryland
  25:1,  # Massachusetts
  26:2,  # Michigan
  27:2,  # Minnesota
  28:3,  # Mississippi
  29:2,  # Missouri
  30:4,  # Montana
  31:2,  # Nebraska
  32:4,  # Nevada
  33:1,  # New Hampshire
  34:1,  # New Jersey
  35:4,  # New Mexico
  36:1,  # New York
  37:3,  # North Carolina
  38:2,  # North Dakota
  39:2,  # Ohio
  40:3,  # Oklahoma
  41:4,  # Oregon
  42:1,  # Pennsylvania
  # 43 is Puerto Rico
  44:1,  # Rhode Island
  45:3,  # South Carolina
  46:2,  # South Dakota
  47:3,  # Tennessee
  48:3,  # Texas
  49:4,  # Utah
  50:1,  # Vermont
  51:3,  # Virginia
  # 52 is Virgin Islands
  53:4,  # Washington
  54:3,  # West Virginia
  55:2,  # Wisconsin
  56:4,  # Wyoming
  3:4,   # American Samoa
  60:4,  # American Samoa
  81:4,  # Baker Island
  7:3,   # Canal Zone
  64:4,  # Federated States of Micronesia
  14:4,  # Guam
  66:4,  # Guam
  84:4,  # Howland Island
  86:4,  # Jarvis Island
  67:4,  # Johnston Atoll
  89:4,  # Kingman Reef
  68:4,  # Marshall Islands
  71:4,  # Midway Islands
  74:4,  # Minor Outlying Islands
  76:3,  # Navassa Island
  69:4,  # Northern Mariana Islands
  70:4,  # Palau
  95:4,  # Palmyra Atoll
  43:3,  # Puerto Rico
  72:3,  # Puerto Rico
  52:3,  # Virgin Islands
  78:3,  # Virgin Islands
  79:4}  # Wake Island
st60_to_name1 = {
  # New England
   1: "Maine",
   3: "New Hampshire",
   4: "Vermont",
   2: "Massachusetts",
   5: "Rhode Island",
   6: "Connecticut",
  # Middle Atlantic
  10: "New York",
  11: "New Jersey",
  13: "Pennsylvania",
  # East North Central
  24: "Ohio",
  23: "Indiana",
  25: "Illinois",
  26: "Michigan",
  22: "Wisconsin",
  # West North Central
  31: "Minnesota",
  32: "Iowa",
  33: "Missouri",
  34: "North Dakota",
  35: "South Dakota",
  36: "Nebraska",
  37: "Kansas",
  # South Atlantic
  41: "Delaware",
  42: "Maryland",
  43: "DC",
  44: "Virginia",
  45: "West Virginia",
  47: "North Carolina",
  46: "South Carolina",
  48: "Georgia",
  49: "Florida",
  # East South Central
  51: "Kentucky",
  52: "Tennessee",
  53: "Alabama",
  54: "Mississippi",
  # West South Central
  65: "Arkansas",
  66: "Louisiana",
  67: "Oklahoma",
  68: "Texas",
  # Mountain
  71: "Montana",
  72: "Idaho",
  73: "Wyoming",
  74: "Colorado",
  75: "New Mexico",
  76: "Arizona",
  77: "Utah",
  78: "Nevada",
  # Pacific
  87: "Washington",
  88: "Oregon",
  89: "California",
  85: "Alaska",
  86: "Hawaii"}
st60_to_name2 = {
  ## New England
  11: "Connecticut",
  # 19 is also Massachusetts, New Hampshire,
  # Rhode Island, and Vermont
  19: "Maine",
  ## Middle Atlantic
  21: "New York",
  22: "New Jersey",
  23: "Pennsylvania",
  ## East North Central
  31: "Ohio",
  32: "Indiana",
  33: "Illinois",
  # 39 also contains Wisconsin
  39: "Michigan",
  ## West North Central
  # 41 also contains Minnesota
  41: "Iowa",
  43: "Missouri",
  # 49 also contains Nebraska, Kansas, and South Dakota
  49: "North Dakota",
  ## South Atlantic
  51: "DC",
  52: "Maryland",
  53: "West Virginia",
  54: "Georgia",
  55: "Florida",
  # 57 also contains South Carolina
  57: "North Carolina",
  # 59 also contains Virginia
  59: "Delaware",
  ## East South Central
  61: "Kentucky",
  62: "Tennessee",
  # 69 also contains Mississippi
  69: "Alabama",
  ## West South Central
  71: "Louisiana",
  72: "Texas",
  # 79 also contains Oklahoma
  79: "Arkansas",
  ## Mountain
  # 81 also contains Colorado and New Mexico
  81: "Arizona",
  # 89 also contains Montana, Nevada, Utah, and Wyoming
  89: "Idaho",
  ## Pacific
  91: "Oregon",
  92: "California",
  # 99 also contains Hawaii and Washington
  99: "Alaska"}
st60_to_name3 = {
  ## New England
  16: "Connecticut",
  14: "Massachusetts",
  # 19 is also New Hampshire,
  # Rhode Island, and Vermont
  19: "Maine",
  ## Middle Atlantic
  21: "New York",
  22: "New Jersey",
  23: "Pennsylvania",
  ## East North Central
  31: "Ohio",
  32: "Indiana",
  33: "Illinois",
  # 39 also contains Wisconsin
  39: "Michigan",
  ## West North Central
  # 49 also contains Iowa, North Dakota, South Dakota,
  # Nebraska, Kansas, and Missouri
  49: "Minnesota",
  ## South Atlantic
  53: "DC",
  56: "North Carolina",
  # 57 also contains Maryland, Virginia, and West Virginia
  57: "Delaware",
  # 58 also contains Georgia
  58: "South Carolina",
  59: "Florida",
  ## East South Central
  # 67 also contains Tennessee
  67: "Kentucky",
  # 69 also contains Mississippi
  69: "Alabama",
  ## West South Central
  72: "Texas",
  # 79 also contains Oklahoma and Louisiana
  79: "Arkansas",
  ## Mountain
  # 89 also contains Idaho, Wyoming, Colorado, New Mexico,
  # Utah, and Nevada
  89: "Montana",
  ## Pacific
  92: "California",
  # 99 also contains Hawaii, Washington, Oregon, and Alaska
  99: "Alaska"}
st60_to_name4 = {
  # New England
  11: "Maine",
  12: "New Hampshire",
  13: "Vermont",
  14: "Massachusetts",
  15: "Rhode Island",
  16: "Connecticut",
  # Middle Atlantic
  21: "New York",
  22: "New Jersey",
  23: "Pennsylvania",
  # East North Central
  31: "Ohio",
  32: "Indiana",
  33: "Illinois",
  34: "Michigan",
  35: "Wisconsin",
  # West North Central
  41: "Minnesota",
  42: "Iowa",
  43: "Missouri",
  44: "North Dakota",
  45: "South Dakota",
  46: "Nebraska",
  47: "Kansas",
  # South Atlantic
  51: "Delaware",
  52: "Maryland",
  53: "DC",
  54: "Virginia",
  55: "West Virginia",
  56: "North Carolina",
  57: "South Carolina",
  58: "Georgia",
  59: "Florida",
  # East South Central
  61: "Kentucky",
  62: "Tennessee",
  63: "Alabama",
  64: "Mississippi",
  # West South Central
  71: "Arkansas",
  72: "Louisiana",
  73: "Oklahoma",
  74: "Texas",
  # Mountain
  81: "Montana",
  82: "Idaho",
  83: "Wyoming",
  84: "Colorado",
  85: "New Mexico",
  86: "Arizona",
  87: "Utah",
  88: "Nevada",
  # Pacific
  91: "Washington",
  92: "Oregon",
  93: "California",
  94: "Alaska",
  95: "Hawaii"}
name_to_fips = {
  # states plus DC
  "Alabama":             1,
  "Alaska":              2,
  "Arizona":             4,
  "Arkansas":            5,
  "California":          6,
  "Colorado":            8,
  "Connecticut":         9,
  "Delaware":           10,
  "DC":                 11,
  "Florida":            12,
  "Georgia":            13,
  "Hawaii":             15,
  "Idaho":              16,
  "Illinois":           17,
  "Indiana":            18,
  "Iowa":               19,
  "Kansas":             20,
  "Kentucky":           21,
  "Louisiana":          22,
  "Maine":              23,
  "Maryland":           24,
  "Massachusetts":      25,
  "Michigan":           26,
  "Minnesota":          27,
  "Mississippi":        28,
  "Missouri":           29,
  "Montana":            30,
  "Nebraska":           31,
  "Nevada":             32,
  "New Hampshire":      33,
  "New Jersey":         34,
  "New Mexico":         35,
  "New York":           36,
  "North Carolina":     37,
  "North Dakota":       38,
  "Ohio":               39,
  "Oklahoma":           40,
  "Oregon":             41,
  "Pennsylvania":       42,
  "Rhode Island":       44,
  "South Carolina":     45,
  "South Dakota":       46,
  "Tennessee":          47,
  "Texas":              48,
  "Utah":               49,
  "Vermont":            50,
  "Virginia":           51,
  "Washington":         53,
  "West Virginia":      54,
  "Wisconsin":          55,
  "Wyoming":            56,
  # territories
  "American Samoa":                   3,
  "American Samoa":                  60,
  "Baker Island":                    81,
  "Canal Zone":                       7,
  "Federated States of Micronesia":  64,
  "Guam":                            14,
  "Guam":                            66,
  "Howland Island":                  84,
  "Jarvis Island":                   86,
  "Johnston Atoll":                  67,
  "Kingman Reef":                    89,
  "Marshall Islands":                68,
  "Midway Islands":                  71,
  "Minor Outlying Islands":          74,
  "Navassa Island":                  76,
  "Northern Mariana Islands":        69,
  "Palau":                           70,
  "Palmyra Atoll":                   95,
  "Puerto Rico":                     43,
  "Puerto Rico":                     72,
  "Virgin Islands":                  52,
  "Virgin Islands":                  78,
  "Wake Island":                     79}
for i in range(1,5):
  temp = eval("st60_to_name{0}".format(i))
  for k in temp:
    if temp[k] not in name_to_fips:
      raise KeyError(
"for st60_to_name{0}: {1} not in name_to_fips".format(i, temp[k]))
print("\nIndex check for dicts is good\n")
base_years = [1964, 1968, 1973, 1976, 1977, 1980, 1988, 2011, 2019]
year_keys = {}
curr_id = 0
curr_year = base_years[curr_id]
next_year = base_years[curr_id + 1]
for i in range(1964, 2024):
  if i == next_year:
    curr_year = next_year
    try:
      curr_id = curr_id + 1
      next_year = base_years[curr_id + 1]
    except:
      next_year = False
  year_keys[i] = curr_year
def read_income_1964(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    if line[0:5] == "00036":
      household.append(try_int(line[10:17]))
      income.append(try_int(line[170:180]))
      state.append(try_int(line[43:45]))
      weight.append(try_int(line[20:30]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight}).dropna()
  temp["state"] = temp["state"].map(st60_to_name1).map(name_to_fips)
  return temp
def read_income_1968(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  curr_household = 0
  for line in f:
    if line[0] == "4":  # family record
      if line[46:48] != "00":       # household record
        curr_household = curr_household + 1
        try_int_trace = 1
        curr_state = try_int(line[36:38])
        try_int_trace = 0
    else:               # person record
      household.append(curr_household)
      income.append(try_int(line[60:66]))
      state.append(curr_state)
      weight.append(try_int(line[204:216]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight}).dropna()
  temp["state"] = temp["state"].map(st60_to_name2).map(name_to_fips)
  return temp
def read_income_1973(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  curr_household = 0
  for line in f:
    if line[0] == "4":  # family record
      if line[46:48] != "00":       # household record
        curr_household = curr_household + 1
        curr_state = try_int(line[36:38])
    else:               # person record
      household.append(curr_household)
      income.append(try_int(line[60:66]))
      state.append(curr_state)
      weight.append(try_int(line[204:216]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight}).dropna()
  temp["state"] = temp["state"].map(st60_to_name3).map(name_to_fips)
  return temp
def read_income_1976(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    type = int(line[6:8])
    if type == 0:                   # household record
      curr_state = int(line[52:54])
    elif 1 <= type and type <= 39:  # person record
      household.append(int(line[0:6]))
      income.append(int(line[246:253]))
      state.append(curr_state)
      weight.append(int(line[117:128]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight})
  temp["state"] = temp["state"].map(st60_to_name3).map(name_to_fips)
  return temp
def read_income_1977(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    type = int(line[6:8])
    if type == 0:                   # household record
      curr_state = int(line[38:40])
    elif 1 <= type and type <= 39:  # person record
      household.append(int(line[0:6]))
      income.append(int(line[246:253]))
      state.append(curr_state)
      weight.append(int(line[117:128]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight})
  temp["state"] = temp["state"].map(st60_to_name4).map(name_to_fips)
  return temp
def read_income_1980(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    type = int(line[6:8])
    if type == 0:                   # household record
      curr_state = int(line[38:40])
    elif 1 <= type and type <= 39:  # person record
      household.append(int(line[0:6]))
      income.append(int(line[247:254]))
      state.append(curr_state)
      weight.append(int(line[117:128]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight})
  temp["state"] = temp["state"].map(st60_to_name4).map(name_to_fips)
  return temp
def read_income_1988(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    if len(line) > 2:
      type = int(line[0])
      if type == 1:    # household record
        curr_state = int(line[39:41])
      elif type == 3:  # person record
        household.append(int(line[1:6]))
        income.append(int(line[439:447]))
        state.append(curr_state)
        weight.append(int(line[65:73]))
  f.close()
  temp = pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight})
  temp["state"] = temp["state"].map(st60_to_name4).map(name_to_fips)
  return temp
def read_income_2011(filename):
  household = []
  income    = []
  state     = []
  weight    = []
  f = open(filename)
  for line in f:
    type = int(line[0])
    if type == 1:    # household record
      curr_state = int(line[41:43])
    elif type == 3:  # person record
      household.append(int(line[1:6]))
      income.append(int(line[579:587]))
      state.append(curr_state)
      weight.append(int(line[154:162]))
  f.close()
  return pd.DataFrame(
    {"household": household,
     "income": income,
     "state": state,
     "weight": weight})
def read_income_2019(h_filename, p_filename):
  household = pd.read_csv(h_filename, header=0)
  person = pd.read_csv(p_filename, header=0)
  temp1 = person[["PH_SEQ", "PTOTVAL", "MARSUPWT"]].rename(
    {"PH_SEQ": "household",
     "PTOTVAL": "income",
     "MARSUPWT": "weight"}, axis=1)
  temp2 = household[["H_SEQ", "GESTFIPS"]].rename(
    {"H_SEQ": "household",
     "GESTFIPS": "state"}, axis=1)
  result = pd.merge(temp1, temp2, how="inner", on="household",
                    sort=False, validate="many_to_one")
  return result
for i in files:
  print("Year {0}".format(i))
  if i < 2019:
    data = eval("read_income_" + str(year_keys[i]))(
      "../data/{0} CPS/{1}".format(i, files[i]))
  else:
    data = read_income_2019("../data/{0} CPS/{1}".format(i, files[i][0]),
                            "../data/{0} CPS/{1}".format(i, files[i][1]))
  data["region"] = data["state"].map(regions)
  temp = data[~((data["region"] >= 1) | (data["region"] <= 4))]
  if len(temp) > 0:
    print("""
Unable to assign regions for some rows;
will drop them. Portion of data getting
dropped:""")
    print(temp)
    print()
  data = data[data["region"].notna()]
  data["state"] = data["state"].astype(int)
  data["region"] = data["region"].astype(int)
  data.sort_index(axis=1, inplace=True)
  data.sort_values("income", inplace=True)
  data.to_csv("data_{0}.txt".format(i), index=False)
print()
## 
##
## End of file `gen_files.py'.
