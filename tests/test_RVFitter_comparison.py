
import unittest

import numpy as np

from RVFitter import RVFitter, RVFitter_comparison
import pkg_resources
import os
import copy
import sys
import matplotlib.pyplot as plt
import pandas as pd

def get_tmp_file(filename):
    with open(filename, "r") as f:
        data = f.read()

    tmp_specsfilelist = filename.replace(".txt", "_tmp.txt")
    with open(tmp_specsfilelist, "w") as f:
        for fileline in data.splitlines():
            tmp_dir = os.path.dirname(filename)
            tmp_data = os.path.join(tmp_dir, fileline)
            f.write(tmp_data + "\n")
    return tmp_specsfilelist

class TestRVFitter_Comparison(unittest.TestCase):
    def setUp(self):
        line_list = pkg_resources.resource_filename(
            "RVFitter",
            "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        self.specsfilelist = pkg_resources.resource_filename(
            "RVFitter", "tests/test_data/debug_specfile_list.txt")
        filename = os.path.join(os.path.dirname(self.specsfilelist),
                                "B275_speclist.pkl")
        self.myfitter = RVFitter.load_from_df_file(filename=filename)

    def test_table(self):
        collected_fitters = {}
        for shape_profile in ["gaussian", "lorentzian"]:
            this_fitter = fit_without_constraints(self.myfitter, shape_profile=shape_profile)
            this_fitter.label = shape_profile + " without constraints"
            collected_fitters["no_constraint_" + shape_profile] = this_fitter

            this_fitter = fit_with_constraints(self.myfitter, shape_profile=shape_profile)
            this_fitter.label = shape_profile + " with constraints"
            collected_fitters["constraint_" + shape_profile] = this_fitter

        #  print(collected_fitterk

        comparer = RVFitter_comparison(dict_of_fitters=collected_fitters)
        comparer.create_overview_df()
        print(comparer.df)

def fit_with_constraints(myfitter, shape_profile="gaussian"):
    # prepare fitting
    this_fitter = copy.deepcopy(myfitter)
    this_fitter.shape_profile = shape_profile
    this_fitter.constrain_parameters(group="cen", constraint_type="epoch")
    this_fitter.constrain_parameters(group="amp", constraint_type="line_profile")
    this_fitter.constrain_parameters(group="sig", constraint_type="line_profile")
    this_fitter.run_fit()
    return this_fitter

def fit_without_constraints(myfitter, shape_profile="gaussian"):
    # prepare fitting
    this_fitter = copy.deepcopy(myfitter)
    this_fitter.shape_profile = shape_profile
    this_fitter.run_fit()
    return this_fitter

if __name__ == "__main__":
    unittest.main()
