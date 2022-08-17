import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import lmfit
import hashlib
import pandas as pd
import astropy.units as u
import astropy.constants as const
import numpy as np
import pickle
import copy
import os
import ipdb

# TODO: add a unique identifier for a line which can be used as key for parameters


class Line(object):
    def __init__(self, line_name, line_profile, wlc_window):
        self.line_name = line_name
        self.line_profile = line_profile
        self.wlc_window = wlc_window

        hash_object = hashlib.md5(self.line_name.encode() +
                                  str(np.random.rand(1)).encode())
        self.hash = hash_object.hexdigest()[:10]
        self._clear()

        self.is_selected = True

    def _clear(self):
        self.normed_wlc = None
        self.normed_flux = None
        self.normed_errors = None
        self.leftValueNorm = None
        self.rightValueNorm = None
        self.leftClip = []
        self.rightClip = []
        self.clipped_wlc = None
        self.clipped_flux = None
        self.clipped_error = None
        self.got_normed = False
        self.is_clipped = False

    def __repr__(self):
        return "Line(\"{name}\", {profile}, {wlc_window})".format(
            name=self.line_name,
            profile=self.line_profile,
            wlc_window=self.wlc_window)

    @property
    def normed_wlc_to_velocity(self):
        if self.normed_wlc is None:
            return None
        else:
            return self.wave_to_vel(self.normed_wlc, self.line_profile)

    @property
    def clipped_error_to_velocity(self):
        if self.clipped_error is None:
            return None
        else:
            return self.wave_to_vel(self.clipped_error, self.line_profile)

    @property
    def clipped_wlc_to_velocity(self):
        if self.clipped_wlc is None:
            return None
        else:
            return self.wave_to_vel(self.clipped_wlc, self.line_profile)

    def wave_to_vel(self, wavelength, w0):
        vel = (const.c.to(u.km / u.s).value * ((wavelength - w0) / w0))
        return vel

    def add_normed_spectrum(self, angstrom, flux, error, leftValueNorm,
                            rightValueNorm):
        self.got_normed = True
        self.spline = splrep([min(angstrom), max(angstrom)],
                             [leftValueNorm, rightValueNorm],
                             k=1)
        self.continuum = splev(angstrom, self.spline)
        self.normed_wlc = angstrom
        self.normed_flux = flux / self.continuum
        if np.isnan(np.sum(self.normed_flux)):
            __import__('ipdb').set_trace()
        self.normed_errors = error
        self.leftValueNorm = leftValueNorm
        self.rightValueNorm = rightValueNorm

        self.clipped_wlc = self.normed_wlc
        self.clipped_flux = self.normed_flux

        masker = (self.clipped_wlc < (self.line_profile + self.wlc_window)) & (
            self.clipped_wlc > (self.line_profile - self.wlc_window))
        # get indices from array of wavelengths

        self.clipped_wlc = self.clipped_wlc[masker]
        self.clipped_flux = self.clipped_flux[masker]

    def clip_spectrum(self, leftClip, rightClip):
        if not self.got_normed:
            print("You cannot clip before you normalize!")
            raise SystemExit
        else:
            if leftClip > rightClip:
                leftClip, rightClip = rightClip, leftClip

            self.leftClip.append(leftClip)
            self.rightClip.append(rightClip)

            masking = (self.clipped_wlc > leftClip) & (self.clipped_wlc <
                                                       rightClip)
            self.clipped_wlc = self.clipped_wlc[~masking]
            self.clipped_flux = self.clipped_flux[~masking]
            self.is_clipped = True

    def plot_clips(self, ax):
        for left, right in zip(self.leftClip, self.rightClip):
            ax.axvspan(left, right, color='blue', alpha=0.2)

    def plot_model(self,
                   df,
                   model_func,
                   result,
                   type,
                   ax=None,
                   plot_dict={"color": "r"},
                   add_legend=False
                   ):
        this_df = df.query('line_hash == @self.hash')
        row = this_df.T.squeeze()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        this_row = copy.deepcopy(row)
        this_row["clipped_wlc_to_velocity"] = np.linspace(
            self.clipped_wlc_to_velocity[0], self.clipped_wlc_to_velocity[-1],
            1000)

        model = model_func(params=result.params, row=this_row, type=type)
        #  print(this_row)
        #  __import__('ipdb').set_trace()
        if add_legend:
            value = int(result.params[this_row["parameters"]["cen"]].value)
            label = "{0} km/s".format(value)
            plot_dict.update({"label": label})

        ax.plot(this_row["clipped_wlc_to_velocity"], model, **plot_dict)

        if add_legend:
            ax.legend()

    def plot_residuals(self,
                       df,
                       model_func,
                       result,
                       type,
                       ax=None,
                       plot_dict={
                           "color": "r",
                           "marker": "."
                       }):
        this_df = df.query('line_hash == @self.hash')
        row = this_df.T.squeeze()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        this_row = copy.deepcopy(row)

        model = model_func(params=result.params, row=this_row, type=type)

        residuals = (this_row["clipped_flux"] -
                     model) / this_row["clipped_error"]

        ax.plot(this_row["clipped_wlc_to_velocity"], residuals, **plot_dict)
        ax.set_ylim(np.abs(max(residuals)) * (-1), np.abs(max(residuals)))

    def plot_clipped_spectrum(self,
                              ax=None,
                              title_prefix=None,
                              plot_velocity=False,
                              plot_dict={
                                  "fmt": 'ro',
                                  "color": 'black',
                                  "ecolor": 'black'
                              }):

        if not self.is_clipped:
            print("Clipped spectrum not found!")
            raise SystemExit
        indices = np.logical_and(
            self.clipped_wlc > self.line_profile - self.wlc_window,
            self.clipped_wlc < self.line_profile + self.wlc_window)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        if self.clipped_error is None:
            print("No errors for normed flux - assuming 1%")
            self.clipped_error = 0.01 * self.clipped_flux
        if plot_velocity:
            xdata = self.clipped_wlc_to_velocity
        else:
            xdata = self.clipped_wlc
        ax.errorbar(xdata[indices],
                    self.clipped_flux[indices],
                    yerr=self.clipped_error[indices],
                    **plot_dict)
        ax.axhline(y=1, linestyle='-.', color='black')
        if plot_velocity:
            ax.set_xlabel("Velocity (km/s)")
        else:
            ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('normalized flux')
        title = self.line_name + ' ' + str(
            "%.0f" % self.line_profile) + ' (clipped)'
        if title_prefix == None:
            ax.set_title(title)
        elif title_prefix == 'no_title':
            pass
        else:
            ax.set_title(title_prefix + title)

    def plot_normed_spectrum(self, ax=None, title_prefix=None):
        if not self.got_normed:
            print("You cannot plot a normed spectrum before adding it!")
            raise SystemExit
        indices = np.logical_and(
            self.normed_wlc > self.line_profile - self.wlc_window,
            self.normed_wlc < self.line_profile + self.wlc_window)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        if self.normed_errors is None:
            print("No errors for normed flux - assuming 1%")
            self.normed_errors = 0.01 * self.normed_flux
        ax.errorbar(self.normed_wlc[indices],
                    self.normed_flux[indices],
                    yerr=self.normed_errors[indices],
                    fmt='ro',
                    color='black',
                    ecolor='black')
        ax.axhline(y=1, linestyle='-.', color='black')
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('flux')
        title = self.line_name + ' ' + str(
            "%.0f" % self.line_profile) + ' (normalized)'
        if title_prefix == None:
            ax.set_title(title)
        else:
            ax.set_title(title_prefix + title)


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

    def __str__(self):
        return 'RVFitter(stars={stars})'.format(stars=self.stars[0].starname)

    def __repr__(self):
        return 'RVFitter(stars={stars})'.format(stars=self.stars)

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
        for i, star in enumerate(self.stars):
            line_ids = []
            res_ids = []
            for j, line in enumerate(star.lines):
                line_ids.append(('{}_{}').format(star.date, line.line_profile))
                res_ids.append(('{}_{}_res').format(star.date,
                                                    line.line_profile))
            dates_list.append(line_ids)
            dates_list.append(res_ids)

        fig = plt.figure(constrained_layout=False,
                         figsize=(4 * len(self.lines), 4 * len(self.stars)))
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

    def plot_fit(self, fig, axes, plot_dict={"zorder": 2.5, "color": "red"}):
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
                               fig,
                               ax_dict,
                               add_legend_label=False,
                               plot_dict={
                                   "zorder": 2.5,
                                   "color": "red"
                               },
                               plot_dict_res={
                                   "marker": ".",
                                   "linestyle": 'None',
                                   "color": "red"
                               }):
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
                                add_legend=add_legend_label,
                                plot_dict=plot_dict)
                line.plot_residuals(df=star.df,
                                    model_func=self.model,
                                    type=self.shape_profile,
                                    result=self.result,
                                    ax=this_ax_res,
                                    plot_dict=plot_dict_res)

                this_ax_res.axhline(0, linestyle="--", color="black")
                # this_ax_res.set_ylim(-0.5, 0.5)

    def plot_data_and_residuals(self,
                                fig,
                                ax_dict,
                                plot_dict={
                                    "fmt": '.',
                                    "color": "black",
                                    "ecolor": "black"
                                }):
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

    def plot_data(self,
                  fig,
                  axes,
                  plot_dict={
                      "fmt": 'o',
                      "color": "black",
                      "ecolor": "black"
                  }):
        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
                this_ax = axes[i, j]
                line.plot_clipped_spectrum(ax=this_ax,
                                           plot_velocity=True,
                                           plot_dict=plot_dict)

    def apply_legend(self, axes):
        for i, star in enumerate(self.stars):
            for j, line in enumerate(star.lines):
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
        with open(output_file, 'w') as file:
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
                height, err_height = round(
                    0.3183099 * amplitude / max(1.e-15, sigma),
                    4), round(0.3183099 * err_amplitude / max(1.e-15, sigma),
                              4)

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
        with open(specsfilelist_name, 'r') as f:
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
            return 1 - amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
        elif type == "lorentzian":
            return 1 - amp * (1 / (1 + ((x - cen) / sigma)**2))
        elif type == "voigt":
            return 1 - (amp * (1 / (sigma * (np.sqrt(2*np.pi)))) * (np.exp(-((x - cen)**2) / ((2 * sigma)**2))) \
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
            shape_func = lambda x, amp, cen, sig, type: __class__.shape(x, amp, cen, sig, type=type)
            return shape_func(x, amp, cen, sig, type=type)
        else:
            ampL = params[par_names["ampL"]]
            sigmaL = params[par_names["sigL"]]
            shape_func = __class__.shape
            return shape_func(x, amp, cen, sig, ampL=ampL, sigmaL=sigmaL, type=type)


class Star(object):
    """Docstring for Star. """
    def __init__(
        self,
        starname,
        date,
        wavelength,  # nm
        flux,
        lines,
        flux_errors=None,
        datetime_formatter="%Y%m%dT%H",
    ):
        self.starname = starname
        self.date = date
        self.wavelength = wavelength
        self.flux = flux
        self.flux_errors = flux_errors
        self.date_time_obj = datetime.strptime(self.date, datetime_formatter)
        self.lines = lines
        self.parameters = {}

    def __repr__(self):
        return "Star: {starname} on {date}".format(starname=self.starname,
                                                   date=self.date)

    def get_line_by_hash(self, hash):
        for line in self.lines:
            if line.hash == hash:
                return line
        else:
            print("Line with name {name} is unknown.".format(name=name))
            print("Known lines are:")
            for line in self.lines:
                print("\t", line.line_name)

    def apply_selecting(self, standard_epoch):
        for line, standard_line in zip(self.lines, standard_epoch.lines):
            line.is_selected = standard_line.is_selected

    @property
    def angstrom(self):
        return self.wavelength * 10

    @classmethod
    def _read_specsfile(cls, specsfile):
        data = np.loadtxt(specsfile)
        wavelength = data[:, 0]
        flux = data[:, 1]
        try:
            flux_errors = data[:, 2]
        except IndexError:
            flux_errors = None
        return wavelength, flux, flux_errors

    @classmethod
    def from_specsfile(cls,
                       starname,
                       date,
                       specsfile,
                       line_list,
                       datetime_formatter="%Y%m%dT%H"):
        """classmethod for creation of Star
        :arg1: starname - name of the star
        :arg2: date - date to be formatted into datetime object
        :arg3: specsfile - path to specsfile which should be read

        :returns: Star
        """

        wavelength, flux, flux_errors = cls._read_specsfile(specsfile)
        lines = cls.make_line_objects(line_list)

        return cls(starname=starname,
                   date=date,
                   wavelength=wavelength,
                   flux=flux,
                   flux_errors=flux_errors,
                   lines=lines,
                   datetime_formatter=datetime_formatter)

    @classmethod
    def from_specsfile_flexi(cls,
                             specsfile,
                             line_list,
                             id_func,
                             datetime_formatter="%Y%m%dT%H",
                             debug=False):
        """classmethod for creation of Star
        :arg1: specsfile - path to specsfile which should be read
        :arg2: id_func - function which processes the specsfile string and returns starname and date
        :param debug:

        :returns: Star
        """
        starname, date = id_func(specsfile)

        wavelength, flux, flux_errors = cls._read_specsfile(specsfile)
        lines = cls.make_line_objects(line_list)
        if debug:
            lines = lines[:3]

        return cls(starname=starname,
                   date=date,
                   wavelength=wavelength,
                   flux=flux,
                   flux_errors=flux_errors,
                   lines=lines,
                   datetime_formatter=datetime_formatter)

    def plot_line(self, line, clipping=False, ax=None, title_prefix=None):
        """TODO: Docstring for plot_line.

        :arg1: TODO
        :returns: TODO

        """
        wlc_window, cl, line_name = line.wlc_window, line.line_profile, line.line_name
        a = self.angstrom
        f = self.flux
        e = self.flux_errors
        ind = np.logical_and(a > cl - wlc_window, a < cl + wlc_window)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        if e is not None:
            ax.errorbar(a[ind], f[ind], yerr=e[ind], fmt='-', color='k')
        else:
            ax.plot(a[ind], f[ind], 'k-')
        ax.axhline(y=1, linestyle='-.', color='black')
        ax.axvline(x=cl, linestyle=':', color='black')
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('flux')
        title = line_name + ' ' + str("%.0f" % cl)
        if title_prefix == None:
            ax.set_title(title)
        else:
            ax.set_title(title_prefix + title)

    @classmethod
    def _read_line_list(cls, linelist):
        data = np.loadtxt(linelist, dtype=str)
        line_names = np.array(data[:, 0])
        line_profiles = np.array(data[:, 1]).astype(float)
        wlc_windows = np.array(data[:, 2]).astype(float)

        return line_names, line_profiles, wlc_windows

    @classmethod
    def make_line_objects(cls, linelist):
        line_names, line_profiles, wlc_windows = cls._read_line_list(linelist)
        lines = []
        for name, profile, wlc in zip(line_names, line_profiles, wlc_windows):
            this_line = Line(line_name=name,
                             line_profile=profile,
                             wlc_window=wlc)
            lines.append(this_line)
        return lines

    def make_dataframe(self):
        df = pd.DataFrame()
        dict_for_df = dict()
        dict_for_df["starname"] = self.starname
        dict_for_df["date"] = self.date
        dict_for_df["wavelength"] = [np.array(self.wavelength)]
        dict_for_df["angstrom"] = [np.array(self.angstrom)]
        dict_for_df["flux"] = [np.array(self.flux)]
        dict_for_df["flux_error"] = [self.flux_errors]
        dict_for_df["date_time_obj"] = self.date_time_obj
        for line in self.lines:
            if line.is_selected:
                dict_for_df["line_name"] = line.line_name
                dict_for_df["line_profile"] = line.line_profile
                dict_for_df["wlc_window"] = line.wlc_window
                dict_for_df["normed_wlc"] = [np.array(line.normed_wlc)]
                dict_for_df["normed_flux"] = [np.array(line.normed_flux)]
                dict_for_df["normed_errors"] = [np.array(line.normed_errors)]
                dict_for_df["leftValueNorm"] = line.leftValueNorm
                dict_for_df["rightValueNorm"] = line.rightValueNorm
                dict_for_df["leftClip"] = [np.array(line.leftClip)]
                dict_for_df["rightClip"] = [np.array(line.rightClip)]
                dict_for_df["clipped_wlc"] = [np.array(line.clipped_wlc)]
                dict_for_df["clipped_flux"] = [np.array(line.clipped_flux)]
                dict_for_df["clipped_error"] = [np.array(line.clipped_error)]
                dict_for_df["line_hash"] = line.hash
                dict_for_df["clipped_error_to_velocity"] = [
                    np.array(line.clipped_error_to_velocity)
                ]
                dict_for_df["clipped_wlc_to_velocity"] = [
                    np.array(line.clipped_wlc_to_velocity)
                ]

                this_df = pd.DataFrame.from_dict(dict_for_df)
                df = df.append(this_df)
        self.df = df

    def setup_parameters(self, shape_profile):
        "Step 4. Fit gaussian by using lmfit"
        self.params = lmfit.Parameters()
        "Initialise variables ..."
        l_parameter_names = []
        for _, row in self.df.iterrows():
            d = dict()
            amplitude = 'amp_line_{name}_epoch_{epoch}'.format(
                name=row["line_hash"], epoch=row["date"])

            self.params.add(amplitude, value=0.5,
                            vary=True, min=0)  # , min=0.01,max=1.0
            d["amp"] = amplitude

            cen = 'cen_line_{name}_epoch_{epoch}'.format(name=row["line_hash"],
                                                         epoch=row["date"])
            self.params.add(cen, value=0, vary=True)  # , min=0.01,max=1.0
            d["cen"] = cen
            sig = 'sig_line_{name}_epoch_{epoch}'.format(name=row["line_hash"],
                                                         epoch=row["date"])
            self.params.add(sig, value=50, vary=True)  # , min=0.01,max=1.0
            d["sig"] = sig
            rv_shift = 'rv_shift_line_{name}_epoch_{epoch}'.format(
                name=row["line_hash"], epoch=row["date"])
            self.params.add(
                rv_shift,
                value=200,
                vary=True,
                expr=
                '299792.458*(cen_line_{name}_epoch_{epoch} - {profile:4.2f})/{profile:4.2f}'
                .format(name=row["line_hash"],
                        epoch=row["date"],
                        profile=row["line_profile"]
                        ))  # , min = 0.)  # , min=0.01,max=1.0
            d["rv_shift"] = rv_shift
            fwhm = 'fwhm_line_{name}_epoch{epoch}'.format(
                name=row["line_hash"], epoch=row["date"])
            self.params.add(
                fwhm,
                value=2.0,
                vary=True,
                expr='2.3548200*sig_line_{name}_epoch_{epoch}'.format(
                    name=row["line_hash"], epoch=row["date"]))
            d["fwhm"] = fwhm

            if shape_profile == "voigt":
                amplitudeL = 'ampL_line_{name}_epoch_{epoch}'.format(
                    name=row["line_hash"], epoch=row["date"])

                self.params.add(amplitudeL, value=0.5,
                                vary=True)  # , min=0.01,max=1.0
                d["ampL"] = amplitudeL

                sigL = 'sigL_line_{name}_epoch_{epoch}'.format(name=row["line_hash"],
                                                             epoch=row["date"])
                self.params.add(sigL, value=50, vary=True)  # , min=0.01,max=1.0
                d["sigL"] = sigL
                rv_shiftL = 'rv_shiftL_line_{name}_epoch_{epoch}'.format(
                    name=row["line_hash"], epoch=row["date"])
                self.params.add(
                    rv_shiftL,
                    value=200,
                    vary=True,
                    expr=
                    '299792.458*(cen_line_{name}_epoch_{epoch} - {profile:4.2f})/{profile:4.2f}'
                    .format(name=row["line_hash"],
                            epoch=row["date"],
                            profile=row["line_profile"]
                            ))  # , min = 0.)  # , min=0.01,max=1.0
                d["rv_shiftL"] = rv_shiftL
                fwhmL = 'fwhmL_line_{name}_epoch{epoch}'.format(
                    name=row["line_hash"], epoch=row["date"])
                self.params.add(
                    fwhmL,
                    value=2.0,
                    vary=True,
                    expr='2.3548200*sig_line_{name}_epoch_{epoch}'.format(
                        name=row["line_hash"], epoch=row["date"]))
                d["fwhmL"] = fwhmL
            l_parameter_names.append(d)

        if "parameters" in self.df.columns:
            del self.df["parameters"]
        self.df["parameters"] = l_parameter_names


class RVFitter_comparison(object):
    """Docstring for RVFitter_comparison. """
    def __init__(self, list_of_fitters, output_folder):
        self.list_of_fitters = list_of_fitters
        stars = [fitter.star for fitter in list_of_fitters]
        # check if all elements of stars are equal
        if len(set(stars)) != 1:
            raise ValueError("All stars must be equal")
        self.star = stars[0]
        self.output_folder = output_folder

    def create_overview_df(self):
        df = pd.DataFrame()
        for fitter in self.list_of_fitters:
            if fitter.constraints_applied:
                suffix = fitter.shape_profile + "_with_constraints"
            else:
                suffix = fitter.shape_profile + "_without_constraints"
            df = self.add_parameters_to_df(df=df,
                                           fitter=fitter,
                                           suffix="_" + suffix)
        self.df = df

    def add_parameters_to_df(self, df, fitter, suffix):
        l_amp = []
        l_cen = []
        l_sig = []
        l_error_amp = []
        l_error_cen = []
        l_error_sig = []
        for parameter in fitter.df["parameters"]:
            amp = parameter["amp"]
            cen = parameter["cen"]
            sig = parameter["sig"]

            l_amp.append(fitter.result.params[amp].value)
            l_cen.append(fitter.result.params[cen].value)
            l_sig.append(fitter.result.params[sig].value)

            l_error_amp.append(fitter.result.params[amp].stderr)
            l_error_cen.append(fitter.result.params[cen].stderr)
            l_error_sig.append(fitter.result.params[sig].stderr)

        this_df = pd.DataFrame({
            "date" + suffix:
            fitter.df["date"],
            "line_profile" + suffix:
            fitter.df["line_profile"],
            "amp" + suffix:
            l_amp,
            "cen" + suffix:
            l_cen,
            "sig" + suffix:
            l_sig,
            "error_amp" + suffix:
            l_error_amp,
            "error_cen" + suffix:
            l_error_cen,
            "error_sig" + suffix:
            l_error_sig
        })

        for col in df.columns:
            if "date" in col:
                check = df[col] == this_df["date" + suffix]
                if np.sum(check) != len(check):
                    raise Exception("Error: date mismatch")
                df["date"] = df[col]
            if "line_profile" in col:
                check = df[col] == this_df["line_profile" + suffix]
                if np.sum(check) != len(check):
                    raise Exception("Error: line_profile mismatch")
                df["line_profile"] = df[col]
        df = pd.concat([df, this_df], axis=1)
        return df

    def compare_fit_results(self, filename, variable, fig_and_ax=None):
        if variable not in ["amp", "cen", "sig"]:
            raise Exception("Error: variable not in ['amp', 'cen', 'sig']")

        if fig_and_ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = fig_and_ax

        columns_to_plot  = [col for col in self.df.columns if col.startswith(variable)]
        #  print(columns_to_plot)
        dates = self.df["date"].unique()
        for date in dates:
            this_df = self.df[self.df["date"] == date]

            p = ax.plot(this_df[variable + '_gaussian_without_constraints'],
                        this_df[variable + '_lorentzian_without_constraints'],
                        'o',
                        label=date + " (no constraints)")
            color = p[0].get_color()
            ax.plot(this_df[variable + '_gaussian_with_constraints'],
                    this_df[variable + '_lorentzian_with_constraints'],
                    'x',
                    color=color,
                    label=date + " (with constraints)")
        mins = self.df[columns_to_plot].min()
        maxes = self.df[columns_to_plot].max()

        ax.plot(np.linspace(np.min(mins), np.max(maxes), 1000),
                np.linspace(np.min(mins), np.max(maxes), 1000),
                '--',
                label="1:1")
        ax.set_xlabel(variable + ' gaussian')
        ax.set_ylabel(variable + ' lorentzian')
        ax.legend()

        fig.savefig(filename)

    def plot_fits_and_residuals(self, color_dict={0: "red", 1: "blue", 2: "green", 3: "orange", 4: "purple", 5: "black"}):
        fig, ax_dict = self.list_of_fitters[0].get_fig_and_ax_dict()

        for idx, this_fitter in enumerate(self.list_of_fitters):
            if idx == 0:
                this_fitter.plot_data_and_residuals(fig=fig, ax_dict=ax_dict)
            this_fitter.plot_fit_and_residuals(fig=fig,
                                               ax_dict=ax_dict,
                                               add_legend_label=True,
                                               plot_dict={"zorder": 2.5,
                                                          "markersize": "1",
                                                          "color": color_dict[idx],
                                                          },
                                               plot_dict_res={
                                                              "marker": ".",
                                                              "linestyle": "None",
                                                              "color": color_dict[idx],
                                                              "markersize": "2"})
        handles, labels = ax_dict[list(ax_dict.items())[0][0]].get_legend_handles_labels()
        labels = [this_fitter.label for this_fitter in self.list_of_fitters]
        fig.legend(handles, labels, ncol=2, loc='lower center')

    def plot_fits_and_data(self, color_dict = {0: "red", 1: "blue", 2: "green", 3: "orange"}, filename=None):
        fig, axes = self.list_of_fitters[0].get_fig_and_axes()
        for idx, this_fitter in enumerate(self.list_of_fitters):
            if idx == 0:
                this_fitter.plot_data(fig=fig, axes=axes)
            this_fitter.plot_fit(fig=fig, axes=axes, plot_dict={"zorder": 2.5, "color": color_dict[idx], "label": this_fitter.label})
        handles, labels = axes[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc='lower center')
        if filename is not None:
            fig.savefig(filename)
