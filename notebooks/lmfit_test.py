# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Testing lmfit

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt


#Define the Gaussian function
def gauss(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))


amp = 10
cen = 5
sigma = 5
x = np.linspace(-5, 15, 100)
model = gauss(x, amp=amp, cen=cen, sigma=sigma)
plt.plot(x, model)

noise = np.random.uniform(low=-1, high=1, size=len(x))
data = model + noise

plt.plot(x, data, linestyle='None', marker="x")

# # Fitting to data

# +
import pkg_resources
import RVFitter
import glob
import os

line_list = pkg_resources.resource_filename(
    "RVFitter",
    "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
pattern = pkg_resources.resource_filename("RVFitter",
                                          "tests/test_data/*.nspec")
specsfilelist = glob.glob(pattern)
print(specsfilelist)

def id_func(specsfile):
    filename = os.path.basename(specsfile)
    splitted_file = filename.split("_")
    starname = splitted_file[0]
    date = splitted_file[2]
    return starname, date


myfitter = RVFitter.RVFitter.from_specsfilelist_flexi(
    specsfilelist=specsfilelist, id_func=id_func, line_list=line_list)
for rvobject in myfitter.rvobjects:
    for line in rvobject.lines:
        angstrom, flux, error = rvobject.angstrom, rvobject.flux, rvobject.flux_errors
        line.add_normed_spectrum(angstrom=angstrom,
                                 flux=flux,
                                 error=error,
                                 leftValueNorm=0.95,
                                 rightValueNorm=1.01)

        left = line.line_profile - 5
        right = line.line_profile + 5
        line.clip_spectrum(leftClip=left, rightClip=right)

myfitter.create_df()
myfitter.setup_parameters()
myfitter.df
# -

import lmfit
myfitter.df["line_hash"].unique()

this_df = myfitter.df[myfitter.df["line_hash"] == 'f546c519ed']


def create_parameters(df):
    suffix = "_line_{name}_epoch_{epoch}"
    rv_shift_base = 'rv_shift'
    sig_base = 'sig'
    fwhm_base = 'fwhm'
    cen_base = 'cen'
    amp_base = 'amp'
    l_parameter_bases = [rv_shift_base, sig_base, fwhm_base, cen_base, amp_base]
    l_parameter_bases_with_suffix = [i + suffix for i in l_parameter_bases]
    defaults = dict()
    defaults["amp"] = 0.5
#     defaults["cen"] = None #line profile
    defaults["rv_shift"] = 200
    defaults["sig"] = 0.3
    defaults["fwhm"] = 2.0
    params = lmfit.Parameters()
    for index, row in df.iterrows():
        for base, base_with_suffix in zip(l_parameter_bases, l_parameter_bases_with_suffix):
            parameter_name = base_with_suffix.format(name=row["line_hash"], epoch=row["date"])
            if base in defaults.keys():
                params.add(name=str(parameter_name), value=defaults[base], vary=True)
            elif base == "cen":
                params.add(name=str(parameter_name), value=row["line_profile"], vary=True)
                
    return params
create_parameters(df=myfitter.df)

fit


