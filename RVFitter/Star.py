
from datetime import datetime

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RVFitter.Line import Line


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
        self.df = None
        self.params = None

    def __repr__(self):
        return "Star: {starname} on {date}".format(starname=self.starname,
                                                   date=self.date)

    def get_line_by_hash(self, hash):
        for line in self.lines:
            if line.hash == hash:
                return line
        raise ValueError(f"No line with hash {hash} found.")

    def get_line(self, line_name, line_profile):
        for line in self.lines:
            if line.line_name == line_name and line.line_profile == line_profile:
                return line
        raise ValueError(f"No line with name {line_name} and profile {line_profile} found.")

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

    def plot_line(self, line, ax=None, title_prefix=None):
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
            _, ax = plt.subplots()

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
        self.params = lmfit.Parameters()
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
                expr='299792.458*(cen_line_{name}_epoch_{epoch} - {profile:4.2f})/{profile:4.2f}'
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
                # , min=0.01,max=1.0
                self.params.add(sigL, value=50, vary=True)
                d["sigL"] = sigL
                rv_shiftL = 'rv_shiftL_line_{name}_epoch_{epoch}'.format(
                    name=row["line_hash"], epoch=row["date"])
                self.params.add(
                    rv_shiftL,
                    value=200,
                    vary=True,
                    expr='299792.458*(cen_line_{name}_epoch_{epoch} - {profile:4.2f})/{profile:4.2f}'
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
