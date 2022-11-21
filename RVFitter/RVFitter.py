import copy
import os
import pickle

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RVFitter.Line import Line
from RVFitter.Star import Star

# TODO: add a unique identifier for a line which can be used as key for parameters


class RVFitter(object):
    """Docstring for RVFitter. """

    def __init__(self, stars, df_name=None, debug=False):
        self.stars = stars
        self.debug = debug
        self.shape_profile = None
        self.constraints_applied = None
        self.label = None
        self.params = None
        self.star = list(set([item.starname for item in self.stars]))

        if len(self.star) != 1:
            print("You seem to mix stars! This does not make sense!")
            raise SystemExit
        else:
            self.star = self.star[0]

        self.lines = self.stars[0].lines

        self.rv_shift_base = 'rv_shift_line_{name}_epoch_{epoch}'
        self.sig_base = 'sig_line_{name}_epoch_{epoch}'
        self.fwhm_base = 'fwhm_line_{name}_epoch{epoch}'
        self.cen_base = 'cen_line_{name}_epoch_{epoch}'
        self.amp_base = 'amp_line_{name}_epoch_{epoch}'
        self.output_file = 'Params_' + self.star + '.dat'
        self.df_name = df_name
        if self.df_name is None:
            self.df = None
        else:
            self.df = pd.read_pickle(self.df_name)
        self.result = None

    def __str__(self):
        return 'RVFitter(stars={stars})'.format(stars=self.stars[0].starname)

    def __repr__(self):
        return 'RVFitter(stars={stars})'.format(stars=self.stars)

    @property
    def subscript(self):
        to_return = ""
        if self.shape_profile == "gaussian":
            to_return += "G"
        elif self.shape_profile == "lorentzian":
            to_return += "L"
        elif self.shape_profile == "voigt":
            to_return += "V"

        if self.constraints_applied:
            to_return += "wc"
        else:
            to_return += "nc"
        return to_return

    def create_df(self, make=True):
        l_dfs = []
        for star in self.stars:
            if make:
                star.make_dataframe()
            l_dfs.append(star.df)
        self.df = pd.concat(l_dfs, axis=0)

    def sort_by_date(self):
        """
        sorts the stars by their date
        """
        dates = [rvo.date for rvo in self.stars]
        sorted_index = np.argsort(dates)
        self.stars = [self.stars[i] for i in sorted_index]

    @classmethod
    def _read_line_list(cls, linelist):
        data = np.loadtxt(linelist, dtype=str)
        line_names = np.array(data[:, 0])
        line_profiles = np.array(data[:, 1]).astype(float)
        wlc_windows = np.array(data[:, 2]).astype(float)

        return line_names, line_profiles, wlc_windows

    def save_df(self, filename=None):
        if filename is None:
            file_to_write = self.df_name
        else:
            file_to_write = filename
        print('Results saved in: {filename}'.format(filename=file_to_write))
        self.df.to_pickle(file_to_write)

    @classmethod
    def load_from_df_file(cls, filename):
        df = pd.read_pickle(filename)
        return cls.load_from_df(df)

    @classmethod
    def load_from_df(cls, df):
        starnames = df['starname'].unique()
        dates = df['date'].unique()

        query = "(starname == '{star}') & (date == '{date}')"

        stars = []
        for star in starnames:
            for date in dates:
                this_query = query.format(star=star, date=date)
                this_df = df.query(this_query)

                lines = []
                wavelengths = []
                fluxes = []
                flux_errors = []

                for _, row in this_df.iterrows():
                    this_line = Line(line_name=row['line_name'],
                                     line_profile=row['line_profile'],
                                     wlc_window=row['wlc_window'])

                    this_line.normed_wlc = row['normed_wlc']
                    this_line.normed_flux = row['normed_flux']
                    this_line.normed_errors = row['normed_errors']
                    this_line.leftValueNorm = row['leftValueNorm']
                    this_line.rightValueNorm = row['rightValueNorm']
                    this_line.leftClip = row['leftClip']
                    this_line.rightClip = row['rightClip']
                    this_line.clipped_wlc = row['clipped_wlc']
                    this_line.clipped_flux = row['clipped_flux']
                    this_line.clipped_error = row['clipped_error']
                    this_line.hash = row['line_hash']
                    this_line.got_normed = True
                    this_line.is_clipped = True

                    lines.append(this_line)
                    fluxes.append(row["flux"])
                    wavelengths.append(row["wavelength"])
                    flux_errors.append(row["flux_error"])

                if cls.find_if_list_of_arrays_contains_different_arrays(
                        fluxes):
                    print(
                        "WARNING: There are different fluxes for the same line!"
                    )
                    print("         This is not supported by the fitter!")
                    raise SystemExit
                if cls.find_if_list_of_arrays_contains_different_arrays(
                        wavelengths):
                    print(
                        "WARNING: There are different wavelengths for the same line!"
                    )
                    print("         This is not supported by the fitter!")
                    raise SystemExit
                if cls.find_if_list_of_arrays_contains_different_arrays(
                        flux_errors):
                    print(
                        "WARNING: There are different flux errors for the same line!"
                    )
                    print("         This is not supported by the fitter!")
                    raise SystemExit

                this_star = Star(starname=star,
                                 lines=lines,
                                 date=date,
                                 wavelength=wavelengths[0],
                                 flux=fluxes[0],
                                 flux_errors=flux_errors[0],
                                 )
                this_star.df = this_df
                stars.append(this_star)
        this_instance = cls(stars=stars)
        this_instance.df = df
        return this_instance

    def get_fig_and_axes(self):
        fig, axes = plt.subplots(
            len(self.stars),
            len(self.lines),
            figsize=(4 * len(self.lines), 4 * len(self.stars)),
        )
        # change distance between axes in subplot of axes
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        if len(self.stars) == 1:
            axes = axes.reshape((1, len(self.lines)))
        if len(self.lines) == 1:
            axes = axes.reshape((len(self.stars), 1))

        return fig, axes

    def get_fig_and_ax_dict(self):
        dates_list = []
        for star in self.stars:
            line_ids = []
            res_ids = []
            for line in star.lines:
                line_ids.append(('{}_{}').format(star.date, line.line_profile))
                res_ids.append(('{}_{}_res').format(star.date,
                                                    line.line_profile))
            dates_list.append(line_ids)
            dates_list.append(res_ids)

        fig = plt.figure(constrained_layout=False,
                         figsize=(4 * len(self.lines), 4 * len(self.stars)))
        # constrained_layout=True doesn't allow hspace to be 0
        ax_dict = fig.subplot_mosaic(
            dates_list,
            sharex=False,
            gridspec_kw={
                "wspace": 0.3,
                "hspace": 0,
                "height_ratios": [5, 2] * len(self.stars),
                # "width_ratios": [1]*len(self.stars),
            },
        )
        return fig, ax_dict

    def plot_fit(self, axes, plot_dict=None):
        if plot_dict is None:
            plot_dict = {"zorder": 2.5, "color": "red"}
        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
                this_ax = axes[i, j]

                line.plot_model(df=star.df,
                                model_func=self.model,
                                type=self.shape_profile,
                                result=self.result,
                                ax=this_ax,
                                plot_dict=plot_dict)

    def plot_fit_and_residuals(self,
                               ax_dict,
                               add_legend_label=False,
                               add_legend_model=False,
                               plot_dict=None,
                               plot_dict_res=None):
        if plot_dict is None:
            plot_dict = {"zorder": 2.5, "color": "red"}
        if plot_dict_res is None:
            plot_dict_res = {
                "marker": ".",
                "linestyle": 'None',
                "color": "red"
            }

        for star in self.stars:
            for line in star.lines:
                this_ax = ax_dict[('{}_{}').format(star.date,
                                                   line.line_profile)]
                this_ax_res = ax_dict[('{}_{}_res').format(
                    star.date, line.line_profile)]

                line.plot_model(df=star.df,
                                model_func=self.model,
                                type=self.shape_profile,
                                result=self.result,
                                ax=this_ax,
                                add_legend_param=add_legend_label,
                                add_legend_model=add_legend_model,
                                plot_dict=plot_dict)
                line.plot_residuals(df=star.df,
                                    model_func=self.model,
                                    type=self.shape_profile,
                                    result=self.result,
                                    ax=this_ax_res,
                                    plot_dict=plot_dict_res)

                this_ax_res.axhline(0, linestyle="--", color="black")

    def plot_data_and_residuals(self,
                                ax_dict,
                                plot_dict=None
                                ):
        if plot_dict is None:
            plot_dict = {"fmt": '.', "color": "black", "ecolor": "black"}
        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
                if i != 0:
                    title_prefix = 'no_title'
                else:
                    title_prefix = None
                this_ax = ax_dict[('{}_{}').format(star.date,
                                                   line.line_profile)]
                this_ax_res = ax_dict[('{}_{}_res').format(
                    star.date, line.line_profile)]
                line.plot_clipped_spectrum(ax=this_ax,
                                           plot_velocity=True,
                                           title_prefix=title_prefix,
                                           plot_dict=plot_dict)
                this_ax_res.set_xlabel("Velocity (km/s)")
                if j == 0:
                    this_ax.text(-0.5, 0.5, star.date,
                                 horizontalalignment='right',
                                 verticalalignment='center',
                                 rotation='vertical', transform=this_ax.transAxes, fontsize=14)

    def plot_data(self,
                  axes,
                  plot_dict=None
                  ):
        if plot_dict is None:
            plot_dict = {"fmt": '.', "color": "black", "ecolor": "black"}
        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
                this_ax = axes[i, j]
                line.plot_clipped_spectrum(ax=this_ax,
                                           plot_velocity=True,
                                           plot_dict=plot_dict)

    def apply_legend(self, axes):
        for i, star in enumerate(self.stars):
            for j in range(len(star.lines)):
                this_ax = axes[i, j]
                this_ax.legend(loc='upper right')

    def plot_model_and_data(self):
        fig, axes = plt.subplots(
            len(self.stars),
            len(self.lines),
            figsize=(4 * len(self.lines), 4 * len(self.stars)),
        )
        # change distance between axes in subplot of axes
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        if len(self.stars) == 1:
            axes = axes.reshape((1, len(self.lines)))

        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
                this_ax = axes[i, j]
                line.plot_clipped_spectrum(ax=this_ax,
                                           plot_velocity=True,
                                           plot_dict={
                                               "fmt": 'k.',
                                               "color": 'black',
                                               "ecolor": 'black'
                                           })
                line.plot_model(df=star.df,
                                model_func=self.model,
                                result=self.result,
                                type=self.shape_profile,
                                ax=this_ax,
                                plot_dict={
                                    "zorder": 2.5,
                                    "color": 'red'
                                })
                # this_ax.set_xlabel("Velocity (km/s)")

    @classmethod
    def find_if_list_of_arrays_contains_different_arrays(cls, list_of_arrays):
        """
        returns True if any of the arrays in the list are different
        """
        if len(list_of_arrays) == 0:
            return False
        else:
            first_array = list_of_arrays[0]
            for array in list_of_arrays:
                if not np.array_equal(first_array, array):
                    return True
            return False

    def setup_parameters(self):
        if self.shape_profile is None:
            raise Exception("shape_profile not set")
        elif self.shape_profile not in ['gaussian', 'lorentzian', 'voigt']:
            raise Exception("shape_profile not recognized")
        else:
            print("shape_profile: {}".format(self.shape_profile))

        for star in self.stars:
            star.setup_parameters(shape_profile=self.shape_profile)
        self.create_df(make=False)
        self.merge_parameters()

    def merge_parameters(self):
        params = lmfit.Parameters()
        for star in self.stars:
            params.update(star.params)
        self.params = params

    def get_parameters(self, group, df=None):
        l_parameter = []
        if df is None:
            df = self.df
        for _, row in df.iterrows():
            l_parameter.append(row["parameters"][group])
        return l_parameter

    def constrain_parameters(self, group, constraint_type="epoch"):
        # constraint of parameter of the same star and epoch
        if constraint_type == "epoch":
            dates = [star.date for star in self.stars]
            parameters_to_constrain = self.get_parameters(group=group)
            for date in dates:
                this_parameter_group = []
                for par in parameters_to_constrain:
                    if date in par:
                        this_parameter_group.append(par)
                for idx, par in enumerate(this_parameter_group):
                    if idx == 0:
                        par_to_constrain = par
                        continue
                    self.params[par].expr = par_to_constrain

        elif constraint_type == "line_profile":
            line_profiles = self.df["line_profile"].unique()

            for profile in line_profiles:
                this_df = self.df.query("line_profile == {}".format(profile))

                parameters_to_constrain = self.get_parameters(group=group,
                                                              df=this_df)
                for idx, par in enumerate(parameters_to_constrain):
                    if idx == 0:
                        par_to_constrain = par
                        continue
                    self.params[par].expr = par_to_constrain
        else:
            print("constraint_type not supported")
            raise SystemExit

    @staticmethod
    def id_func(specsfile):
        filename = os.path.basename(specsfile)
        splitted_file = filename.split("_")
        starname = splitted_file[0]
        date = splitted_file[2]
        return starname, date

    def print_fit_result(self, output_file=None):
        if output_file is None:
            output_file = self.output_file

        print("Write parameters to file: {filename}".format(
            filename=output_file))

        for _, row in self.df.iterrows():
            line_profile = row["line_profile"]
            amplitude, err_amplitude = round(
                self.result.params[row["parameters"]["amp"]].value,
                4), round(
                    self.result.params[row["parameters"]["amp"]].stderr, 4)
            sigma, err_sigma = round(
                self.result.params[row["parameters"]["sig"]].value,
                4), round(
                    self.result.params[row["parameters"]["sig"]].stderr, 4)
            centroid, err_centroid = round(
                self.result.params[row["parameters"]["cen"]].value,
                4), round(
                    self.result.params[row["parameters"]["cen"]].stderr, 4)
            # height, err_height = round(
            #     0.3183099 * amplitude / max(1.e-15, sigma),
            #     4), round(0.3183099 * err_amplitude / max(1.e-15, sigma),
            #                 4)

            print('-----------',
                    row["line_name"] + ' ' + str(line_profile),
                    '-----------')
            print('Amplitude= ', '\t', amplitude, ' +/-\t', err_amplitude)
            print('Sigma=     ', '\t', sigma, ' +/-\t', err_sigma)
            print('Centroid=     ', '\t', centroid, ' +/-\t', err_centroid)
            if self.shape_profile == "voigt":
                amplitudeL, err_amplitudeL = round(
                    self.result.params[row["parameters"]["ampL"]].value,
                    4), round(
                        self.result.params[row["parameters"]["ampL"]].stderr, 4)
                sigmaL, err_sigmaL = round(
                    self.result.params[row["parameters"]["sigL"]].value,
                    4), round(
                        self.result.params[row["parameters"]["sigL"]].stderr, 4)
                centroidL, err_centroidL = round(
                    self.result.params[row["parameters"]["cenL"]].value,
                    4), round(
                        self.result.params[row["parameters"]["cenL"]].stderr, 4)
                print('AmplitudeL= ', '\t', amplitudeL, ' +/-\t',
                        err_amplitudeL)
                print('SigmaL=     ', '\t', sigmaL, ' +/-\t', err_sigmaL)
                print('CentroidL=     ', '\t', centroidL, ' +/-\t',
                        err_centroidL)

    def run_fit(self, objective=None):
        if objective is None:
            this_objective = self.objective
        else:
            this_objective = objective
        self.result = lmfit.minimize(this_objective,
                                     self.params,
                                     args=([self.df]),
                                     kws={"type": self.shape_profile})

    def save_fit_result(self, filename):
        dict_to_dump = {
            "result": self.result,
            "label": self.label,
            "shape_profile": self.shape_profile,
            "constraints_applied": self.constraints_applied,
        }
        with open(filename, "wb") as f:
            pickle.dump(dict_to_dump, f)
        print("Fit result saved to file: {filename}".format(filename=filename))

    def load_fit_result(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.result = data["result"]
        self.label = data["label"]
        self.shape_profile = data["shape_profile"]
        self.constraints_applied = data["constraints_applied"]

        print(
            "Load fit result from file: {filename}".format(filename=filename))

    def fit_with_constraints(self, shape_profile="gaussian"):
        # prepare fitting
        this_fitter = copy.deepcopy(self)
        this_fitter.shape_profile = shape_profile
        this_fitter.setup_parameters()
        this_fitter.label = shape_profile + " with constraints"
        this_fitter.constraints_applied = True
        this_fitter.constrain_parameters(group="cen", constraint_type="epoch")
        this_fitter.constrain_parameters(group="amp",
                                         constraint_type="line_profile")
        this_fitter.constrain_parameters(group="sig",
                                         constraint_type="line_profile")
        if shape_profile == "voigt":
            this_fitter.constrain_parameters(group="ampL",
                                             constraint_type="line_profile")
            this_fitter.constrain_parameters(group="sigL",
                                             constraint_type="line_profile")
        this_fitter.run_fit()
        return this_fitter

    def fit_without_constraints(self, shape_profile="gaussian"):
        # prepare fitting
        this_fitter = copy.deepcopy(self)
        this_fitter.shape_profile = shape_profile
        this_fitter.setup_parameters()
        this_fitter.label = shape_profile + " without constraints"
        this_fitter.constraints_applied = False
        this_fitter.run_fit()
        return this_fitter

    def get_df_from_star(self, name, date):
        query = "starname == '{name}' & date == '{date}'".format(name=name,
                                                                 date=date)
        this_df = self.df.query(query)
        if len(this_df) == 0:
            print("No data for star {name} on {date}".format(name=name,
                                                             date=date))
        return this_df

    @classmethod
    def from_specsfilelist_name_flexi(cls,
                                      specsfilelist_name,
                                      line_list,
                                      id_func=None,
                                      share_line_hashes=False,
                                      datetime_formatter="%Y%m%dT%H",
                                      debug=False):
        with open(specsfilelist_name, 'r', encoding='utf-8') as f:
            specsfilelist = f.read().splitlines()
        if debug:
            specsfilelist = specsfilelist[:2]

        if id_func is None:
            id_func = __class__.id_func

        stars = [
            Star.from_specsfile_flexi(specsfile=specsfile,
                                      id_func=id_func,
                                      datetime_formatter=datetime_formatter,
                                      line_list=line_list,
                                      debug=debug)
            for specsfile in specsfilelist
        ]

        if share_line_hashes:
            line_hashes = [star.lines[0].hash for star in stars]
            for star in stars:
                for line, hash in zip(star.lines, line_hashes):
                    line.hash = hash
            stars = [star for star in stars]
        return cls(stars=stars)

    @classmethod
    def from_specsfilelist_flexi(cls,
                                 specsfilelist,
                                 id_func,
                                 line_list,
                                 datetime_formatter="%Y%m%dT%H",
                                 debug=False):
        stars = [
            Star.from_specsfile_flexi(specsfile=specsfile,
                                      id_func=id_func,
                                      datetime_formatter=datetime_formatter,
                                      line_list=line_list,
                                      debug=debug)
            for specsfile in specsfilelist
        ]
        return cls(stars=stars)

    @staticmethod
    def objective(params, df, type="lorentzian"):
        """ calculate total residual for fits to several data sets held
        in a 2-D array, and modeled by Gaussian functions"""
        resid = []
        for _, row in df.iterrows():
            resid.extend(
                (row["clipped_flux"] - __class__.model(params, row, type)))
        return np.array(resid)

    @staticmethod
    def shape(x, amp, cen, sigma, ampL=None, sigmaL=None, type="lorentzian"):
        if type == "gaussian":
            return 1 - amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))
        elif type == "lorentzian":
            return 1 - amp * (1 / (1 + ((x - cen) / sigma) ** 2))
        elif type == "voigt":
            return 1 - (amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cen) ** 2) / ((2 * sigma) ** 2)))
                        + (ampL * sigmaL ** 2 / ((x - cen) ** 2 + sigmaL ** 2)))
            # raise NotImplementedError
        else:
            raise ValueError("Unknown shape type")

    @staticmethod
    def model(params, row, type):
        par_names = row["parameters"]
        amp = params[par_names["amp"]]
        cen = params[par_names["cen"]]
        sig = params[par_names["sig"]]
        x = row["clipped_wlc_to_velocity"]
        if type != "voigt":
            def shape_func(x, amp, cen, sig, type): return __class__.shape(
                x, amp, cen, sig, type=type)
            return shape_func(x, amp, cen, sig, type=type)
        else:
            ampL = params[par_names["ampL"]]
            sigmaL = params[par_names["sigL"]]
            shape_func = __class__.shape
            return shape_func(x, amp, cen, sig, ampL=ampL, sigmaL=sigmaL, type=type)

    def get_results_per_line(self, line_name, line_profile):
        if self.result is None:
            raise ValueError("No results yet. Run fit first.")

        l_dfs = []
        for star in self.stars:
            this_dict = {}
            this_dict["starname"] = [star.starname]
            this_dict["date"] = [star.date]
            this_dict["line_name"] = [line_name]
            this_dict["line_profile"] = [line_profile]

            this_line = star.get_line(line_name, line_profile)
            partial_df = self.df.query('line_hash == @this_line.hash')
            POIs = partial_df["parameters"][0]
            #  for parameter_name, result_name in POIs.items():
            this_dict["cen"] = [self.result.params[POIs["cen"]].value]
            this_dict["amp"] = [self.result.params[POIs["amp"]].value]
            this_dict["sig"] = [self.result.params[POIs["sig"]].value]
            this_df = pd.DataFrame.from_dict(this_dict)
            l_dfs.append(this_df)
        return pd.concat(l_dfs)

    def get_results_per_linelist(self, list_of_name_and_profile_tuples):
        for name, profile in list_of_name_and_profile_tuples:
            print(name, profile)
            this_df = self.get_results_per_line(name, profile)
            if "df" not in locals():
                df = this_df
            else:
                df = pd.concat([df, this_df])
        return df

