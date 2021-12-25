import unittest
import pkg_resources
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RVFitter import RVFitter


def id_func(specsfile):
    filename = os.path.basename(specsfile)
    splitted_file = filename.split("_")
    starname = splitted_file[0]
    date = splitted_file[2]
    return starname, date

def objective(params, df):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Gaussian functions"""
    resid = []
    for _, row in df.iterrows():
        resid.extend((row["clipped_flux"] -
                          gauss_dataset(params, row)))
    return np.array(resid)

def gauss(x, amp, cen, sigma):
    #    "basic gaussian"
    #    return 1-amp*np.exp(-(x-cen)**2/(2.*sigma**2))
    "basic lorentzian"
    return 1 - (amp / (1 + ((1.0 * x - cen) / sigma) ** 2)) / (np.pi * sigma)

def gauss_dataset(params, row):
    par_names = row["parameters"]
    amp = params[par_names["amp"]]
    cen = params[par_names["cen"]]
    sig = params[par_names["sig"]]
    x = row["clipped_wlc"]
    return gauss(x, amp, cen, sig)



class TestRVFitter(unittest.TestCase):
    line_list = pkg_resources.resource_filename(
        "RVFitter", "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
    pattern = pkg_resources.resource_filename("RVFitter",
                                              "tests/test_data/*.nspec")
    specsfilelist = glob.glob(pattern)

    myfitter = RVFitter.from_specsfilelist_flexi(specsfilelist=specsfilelist,
                                                 id_func=id_func,
                                                 line_list=line_list)

    def test_fitting(self):
        line_list = pkg_resources.resource_filename(
            "RVFitter",
            "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        pattern = pkg_resources.resource_filename("RVFitter",
                                                  "tests/test_data/*.nspec")
        specsfilelist = glob.glob(pattern)

        myfitter = RVFitter.from_specsfilelist_flexi(
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
        myfitter.constrain_parameters(group="sig")
        #  myfitter.constrain_parameters(group="amp")
        myfitter.set_objective(objective)
        myfitter.run_fit()
        myfitter.print_fit_result()


        #  for _, row in myfitter.df.iterrows():
        #      plt.plot(row["clipped_wlc"], row["clipped_flux"])
        #      plt.plot(row["clipped_wlc"], gauss(row["clipped_wlc"]))
        #      plt.show()


if __name__ == "__main__":
    unittest.main()
