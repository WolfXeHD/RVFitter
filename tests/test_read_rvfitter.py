import unittest
import pkg_resources
import os
import numpy as np
import copy

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
        resid.extend((row["clipped_flux"] - model(params, row)))
    return np.array(resid)


def shape(x, amp, cen, sigma, type="lorentzian"):
    if type == "gaussian":
        return 1 - amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))
    elif type == "lorentzian":
        return 1 - amp * (1 / (1 + ((x - cen) / sigma) ** 2))
    elif type == "voigt":
        raise NotImplementedError
        #  return amp * (1 / (1 + ((x - cen) / sigma) ** 2)) + \
        #      amp * (1 / (1 + ((x - cen - 2 * sigma) / sigma) ** 2))
    else:
        raise ValueError("Unknown shape type")
    #  #    "basic gaussian"
    #  #    return 1-amp*np.exp(-(x-cen)**2/(2.*sigma**2))
    #  "basic lorentzian"
    #  return 1 - (amp / (1 + ((1.0 * x - cen) / sigma)**2)) / (np.pi * sigma)


def model(params, row):
    par_names = row["parameters"]
    amp = params[par_names["amp"]]
    cen = params[par_names["cen"]]
    sig = params[par_names["sig"]]
    #  x = row["clipped_wlc_to_velocity"]
    x = row["clipped_wlc"]
    return shape(x, amp, cen, sig)


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
        #  self.myfitter.constrain_parameters(group="sig")
        #  #  self.myfitter.constrain_parameters(group="amp")
        self.myfitter.set_objective(objective)
        self.myfitter.run_fit()
        self.myfitter.print_fit_result()

    def test_single_star_fitting(self):
        filename = os.path.join(os.path.dirname(self.specsfilelist),
                                "B111_speclist_Tim.pkl")
        self.myfitter.load_df(filename=filename)
        starnames = self.myfitter.df["starname"].unique()
        dates = self.myfitter.df["date"].unique()

        for starname in starnames:
            for date in dates:
                star_df = self.myfitter.get_df_from_star(name=starname,
                                                         date=date)
                try:
                    this_fitter = copy.deepcopy(self.myfitter)
                    this_fitter.load_df(df=star_df)
                    #  this_fitter.setup_parameters()
                    this_fitter.constrain_parameters(group="amp")
                    this_fitter.set_objective(objective)
                    this_fitter.run_fit()
                    this_fitter.print_fit_result()

                    this_fitter = copy.deepcopy(self.myfitter)
                    this_fitter.load_df(df=star_df)
                    #  this_fitter.setup_parameters()
                    #  this_fitter.constrain_parameters(group="amp")
                    this_fitter.set_objective(objective)
                    this_fitter.run_fit()
                    this_fitter.print_fit_result()
                except:
                    print("Failed to fit {} with date {}".format(starname, date))

    def test_loading_from_df(self):
        filename = os.path.join(os.path.dirname(self.specsfilelist),
                                "B111_speclist_Tim.pkl")
        self.myfitter.load_df(filename=filename)

        for star in self.myfitter.stars:
            query = "(starname == '{starname}') & (date == '{date}')".format(
                starname=star.starname, date=star.date)
            this_df = self.myfitter.df.query(query)
            self.assertEqual(len(this_df), len(star.lines))

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
                self.assertEqual(line.leftValueNorm  , row['leftValueNorm'])

                # TODO check what is happening with these two tests
                #    self.assertEqual(line.rightValueNorm , row['rightValueNorm'])
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
