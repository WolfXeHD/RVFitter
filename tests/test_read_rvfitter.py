import unittest
import pkg_resources
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipdb

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
        resid.extend((row["clipped_flux"] - gauss_dataset(params, row)))
    return np.array(resid)


def gauss(x, amp, cen, sigma):
    #    "basic gaussian"
    #    return 1-amp*np.exp(-(x-cen)**2/(2.*sigma**2))
    "basic lorentzian"
    return 1 - (amp / (1 + ((1.0 * x - cen) / sigma)**2)) / (np.pi * sigma)


def gauss_dataset(params, row):
    par_names = row["parameters"]
    amp = params[par_names["amp"]]
    cen = params[par_names["cen"]]
    sig = params[par_names["sig"]]
    x = row["clipped_wlc"]
    return gauss(x, amp, cen, sig)


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


class TestRVFitter(unittest.TestCase):
    def setUp(self):
        self.line_list = pkg_resources.resource_filename(
            "RVFitter",
            "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        self.specsfilelist = pkg_resources.resource_filename(
            "RVFitter", "tests/test_data/debug_specfile_list.txt")

        tmp_specsfilelist = get_tmp_file(self.specsfilelist)

        self.myfitter = RVFitter.from_specsfilelist_name_flexi(
            specsfilelist_name=tmp_specsfilelist,
            id_func=id_func,
            line_list=self.line_list)

    def test_fitting(self):
        for star in self.myfitter.stars:
            for line in star.lines:
                angstrom, flux, error = star.angstrom, star.flux, star.flux_errors
                line.add_normed_spectrum(angstrom=angstrom,
                                         flux=flux,
                                         error=error,
                                         leftValueNorm=0.95,
                                         rightValueNorm=1.01)

                left = line.line_profile - 5
                right = line.line_profile + 5
                line.clip_spectrum(leftClip=left, rightClip=right)

        self.myfitter.create_df()
        self.myfitter.setup_parameters()
        self.myfitter.constrain_parameters(group="sig")
        #  self.myfitter.constrain_parameters(group="amp")
        self.myfitter.set_objective(objective)
        self.myfitter.run_fit()
        #  self.myfitter.print_fit_result()

        #  for _, row in myfitter.df.iterrows():
        #      plt.plot(row["clipped_wlc"], row["clipped_flux"])
        #      plt.plot(row["clipped_wlc"], gauss(row["clipped_wlc"]))
        #      plt.show()

    def test_loading_from_df(self):
        filename = os.path.join(os.path.dirname(self.specsfilelist),
                                "B111_speclist_Tim.pkl")
        self.myfitter.load_df(filename=filename)

        for star_idx, star in enumerate(self.myfitter.stars):
            query = "(starname == '{starname}') & (date == '{date}')".format(
                starname=star.starname, date=star.date)
            this_df = self.myfitter.df.query(query)
            #  (self.myfitter.df["starname"] == star.starname)
            #  & (self.myfitter.df["date"] == star.date)]
            self.assertEqual(len(this_df), len(star.lines))
            # iterate over rows in dataframe
            for (_, row), line in zip(this_df.iterrows(), star.lines):
                self.assertEqual(line.line_name, row["line_name"])
                self.assertEqual(line.line_profile, row["line_profile"])
                self.assertEqual(line.wlc_window, row["wlc_window"])
                self.assertEqual(line.normed_wlc.any(),
                                 row['normed_wlc'].any())
                self.assertEqual(line.normed_flux.any(),
                                 row['normed_flux'].any())
                self.assertEqual(line.normed_errors.any(),
                                 row['normed_errors'].any())

                # TODO check what is happening with these two tests
                #  self.assertEqual(line.leftValueNorm  , row['leftValueNorm'])
                #  self.assertEqual(line.rightValueNorm , row['rightValueNorm'])
                self.assertEqual(line.leftClip, row['leftClip'])
                self.assertEqual(line.rightClip, row['rightClip'])
                self.assertEqual(line.clipped_wlc.any(),
                                 row['clipped_wlc'].any())
                self.assertEqual(line.clipped_flux.any(),
                                 row['clipped_flux'].any())
                self.assertEqual(line.clipped_error.any(),
                                 row['clipped_error'].any())
                self.assertEqual(line.hash, row['line_hash'])


if __name__ == "__main__":
    unittest.main()
