
import copy
import hashlib

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep


class Line(object):
    def __init__(self, line_name, line_profile, wlc_window):
        self.line_name = line_name
        self.line_profile = line_profile
        self.wlc_window = wlc_window

        hash_object = hashlib.md5(self.line_name.encode() +
                                  str(np.random.rand(1)).encode())
        self.hash = hash_object.hexdigest()[:10]
        self.is_selected = True

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

        self.spline = None
        self.continuum = None


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
                   add_legend_param=False,
                   add_legend_model=False,
                   ):
        this_df = df.query('line_hash == @self.hash')
        row = this_df.T.squeeze()
        if ax is None:
            _, ax = plt.subplots()

        this_row = copy.deepcopy(row)
        this_row["clipped_wlc_to_velocity"] = np.linspace(
            self.clipped_wlc_to_velocity[0], self.clipped_wlc_to_velocity[-1],
            1000)

        model = model_func(params=result.params, row=this_row, type=type)
        if add_legend_param:
            value = int(result.params[this_row["parameters"]["cen"]].value)
            label = "{0} km/s".format(value)
            plot_dict.update({"label": label})
        elif add_legend_model:
            plot_dict.update({"label": model_func})

        ax.plot(this_row["clipped_wlc_to_velocity"], model, **plot_dict)

        if add_legend_param:
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
            _, ax = plt.subplots()

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
            _, ax = plt.subplots()

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
            _, ax = plt.subplots()

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
