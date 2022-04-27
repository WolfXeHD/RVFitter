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


    def plot_model(self, df, model_func, result, ax=None, plot_dict={"color": "r"}):
        this_df = df.query(
            'line_hash == @self.hash')
        row = this_df.T.squeeze()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax

        this_row = copy.deepcopy(row)
        this_row["clipped_wlc_to_velocity"] = np.linspace(
            self.clipped_wlc_to_velocity[0],
            self.clipped_wlc_to_velocity[-1], 1000)

        model = model_func(params=result.params, row=this_row)

        ax.plot(this_row["clipped_wlc_to_velocity"], model,
                **plot_dict)

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
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('flux')
        title = self.line_name + ' ' + str(
            "%.0f" % self.line_profile) + ' (clipped)'
        if title_prefix == None:
            ax.set_title(title)
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


class RVFitter(lmfit.Model):
    """Docstring for RVFitter. """
    def __init__(self, specsfilelist_name, stars, line_list, debug=False):
        self.specsfilelist_name = specsfilelist_name
        self.stars = stars
        self.line_list = line_list
        self.debug = debug
        self.line_names, self.line_profiles, self.wlc_windows = self._read_line_list(
            self.line_list)
        self.lines = self.make_line_objects()
        #  self.objective = None
        self.objective_set = False
        self.params = None
        self.star = list(set([item.starname for item in self.stars]))
        if len(self.star) != 1:
            print("You seem to mix stars! This does not make sense!")
            raise SystemExit
        else:
            self.star = self.star[0]

        self.rv_shift_base = 'rv_shift_line_{name}_epoch_{epoch}'
        self.sig_base = 'sig_line_{name}_epoch_{epoch}'
        self.fwhm_base = 'fwhm_line_{name}_epoch{epoch}'
        self.cen_base = 'cen_line_{name}_epoch_{epoch}'
        self.amp_base = 'amp_line_{name}_epoch_{epoch}'
        self.df = None

    @property
    def df_name(self):
        if not self.debug:
            return self.specsfilelist_name.replace('.txt', '.pkl')
        else:
            return self.specsfilelist_name.replace('.txt', '_DEBUG.pkl')

    def create_df(self, make=True):
        l_dfs = []
        for star in self.stars:
            if make:
                star.make_dataframe()
            l_dfs.append(star.df)
        self.df = pd.concat(l_dfs, axis=0)

    def set_objective(self, objective):
        self.objective_set = True
        self.objective = objective

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

    def make_line_objects(self):
        lines = []
        for name, profile, wlc in zip(self.line_names, self.line_profiles,
                                      self.wlc_windows):
            this_line = Line(line_name=name,
                             line_profile=profile,
                             wlc_window=wlc)
            lines.append(this_line)
        return lines

    def save_df(self, filename=None):
        if filename is None:
            file_to_write = self.df_name
        else:
            file_to_write = filename
        print('Results saved in: {filename}'.format(filename=file_to_write))
        self.df.to_pickle(file_to_write)

    @classmethod
    def create_fitter_from_df(cls):
        pass

    # TODO: shouldn't this be a classmethod?
    def load_df(self, filename=None, df=None):
        if filename is not None and df is not None:
            print("You cannot load a dataframe and specify a filename!")
            raise SystemExit

        if filename is None:
            file_to_read = self.df_name
        else:
            file_to_read = filename

        if df is not None:
            self.df = df
            print("Using the dataframe you passed in.")
        else:
            self.df = pd.read_pickle(file_to_read)
            print("Loading dataframe from {filename}".format(
                filename=file_to_read))

        starnames = self.df['starname'].unique()
        dates = self.df['date'].unique()

        query = "(starname == '{star}') & (date == '{date}')"

        stars = []
        for star in starnames:
            for date in dates:
                this_query = query.format(star=star, date=date)
                this_df = self.df.query(this_query)

                lines = []
                wavelengths = []
                fluxes = []
                flux_errors = []

                for idx, row in this_df.iterrows():
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

                if self.find_if_list_of_arrays_contains_different_arrays(
                        fluxes):
                    print(
                        "WARNING: There are different fluxes for the same line!"
                    )
                    print("         This is not supported by the fitter!")
                    raise SystemExit
                if self.find_if_list_of_arrays_contains_different_arrays(
                        wavelengths):
                    print(
                        "WARNING: There are different wavelengths for the same line!"
                    )
                    print("         This is not supported by the fitter!")
                    raise SystemExit
                if self.find_if_list_of_arrays_contains_different_arrays(
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
                                 flux_errors=flux_errors[0])
                this_star.df = this_df
                stars.append(this_star)
        self.stars = stars
        self.setup_parameters()

    def plot_model_and_data(self):
        fig, axes = plt.subplots(len(self.stars), len(self.lines),
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
                                               "fmt": 'ko',
                                               "color": 'black',
                                               "ecolor": 'black'
                                           })
                line.plot_model(df=star.df,
                                model_func=self.model,
                                result=self.result,
                                ax=this_ax, plot_dict={"zorder": 2.5, "color": 'red'})


    def find_if_list_of_arrays_contains_different_arrays(self, list_of_arrays):
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
        for star in self.stars:
            star.setup_parameters()
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

                parameters_to_constrain = self.get_parameters(group=group, df=this_df)
                for idx, par in enumerate(parameters_to_constrain):
                    if idx == 0:
                        par_to_constrain = par
                        continue
                    self.params[par].expr = par_to_constrain
        else:
            print("constraint_type not supported")
            raise SystemExit

    def print_fit_result(self):
        output_file = 'Params_' + self.star + '.dat'

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
                fwhm, err_fwhm = round(
                    self.result.params[row["parameters"]["fwhm"]].value,
                    4), round(
                        self.result.params[row["parameters"]["fwhm"]].stderr,
                        4)
                centroid, err_centroid = round(
                    self.result.params[row["parameters"]["cen"]].value,
                    4), round(
                        self.result.params[row["parameters"]["cen"]].stderr, 4)
                #  vsini, err_vsini = round(vsini_constant * fwhm, 4), round(vsini_constant * err_fwhm, 4)
                height, err_height = round(
                    0.3183099 * amplitude / max(1.e-15, sigma),
                    4), round(0.3183099 * err_amplitude / max(1.e-15, sigma),
                              4)
                print('-----------',
                      row["line_name"] + ' ' + str(line_profile),
                      '-----------')
                print('Amplitude= ', '\t', amplitude, ' +/-\t', err_amplitude)
                print('Height= ', '\t', height, ' +/-\t', err_height)
                #            print 'FWHM=     ','\t',fwhm,' +/-\t',err_fwhm
                print('Sigma=     ', '\t', sigma, ' +/-\t', err_sigma)
                print('Centroid=     ', '\t', centroid, ' +/-\t', err_centroid)
                print('RV=     ', '\t',
                      const.c * ((centroid - line_profile) / line_profile),
                      ' +/-\t', (err_centroid / centroid) * const.c)
                #            print 'vsini=        ','\t',vsini,' +/-\t',err_vsini

                file.write('\n')
                file.write('Line profile ' + row["line_name"] + '\t' +
                           str(line_profile) + '\n')
                #            file.write('Amplitude '+'\t'+str(amplitude)+'\t'+str(err_amplitude)+'\n')
                file.write('Amplitude ' + '\t' + str(height) + '\t' +
                           str(err_height) + '\n')
                #            file.write('FWHM '+'\t'+str(fwhm)+'\t'+str(err_fwhm)+'\n')
                file.write('sigma ' + '\t' + str(sigma) + '\t' +
                           str(err_sigma) + '\n')
                file.write('centroid ' + '\t' + str(centroid) + '\t' +
                           str(err_centroid) + '\n')
                file.write('RV_line ' + '\t' + str((
                    (centroid - line_profile) / line_profile) * 299792.458) +
                           '\t' + str((err_centroid / centroid) * 299792.458) +
                           '\n')
            #           file.write('vsini '+'\t'+str(vsini)+'\t'+str(err_vsini)+'\n')

    def run_fit(self):
        if self.objective_set is False:
            print("You did not specify an objective function!")
            raise SystemExit
        # TODO: minimize can also take kwargs --> can be used to set line-shape
        self.result = lmfit.minimize(self.objective,
                                     self.params,
                                     args=([self.df]))

    def save_fit_result(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.result, f)

    def load_fit_result(self, filename):
        with open(filename, "rb") as f:
            self.result = pickle.load(f)

    def plot_fit_result(self):
        pass

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
                                      id_func,
                                      line_list,
                                      datetime_formatter="%Y%m%dT%H",
                                      debug=False):
        with open(specsfilelist_name, 'r') as f:
            specsfilelist = f.read().splitlines()
        if debug:
            specsfilelist = specsfilelist[:2]

        stars = [
            Star.from_specsfile_flexi(specsfile=specsfile,
                                      id_func=id_func,
                                      datetime_formatter=datetime_formatter,
                                      line_list=line_list,
                                      debug=debug)
            for specsfile in specsfilelist
        ]
        return cls(specsfilelist_name=specsfilelist_name,
                   stars=stars,
                   line_list=line_list)

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
        return cls(specsfile=None, stars=stars, line_list=line_list)

    @staticmethod
    def objective(params, df):
        """ calculate total residual for fits to several data sets held
        in a 2-D array, and modeled by Gaussian functions"""
        resid = []
        for _, row in df.iterrows():
            resid.extend((row["clipped_flux"] - __class__.model(params, row)))
        return np.array(resid)

    @staticmethod
    def shape(x, amp, cen, sigma, type="lorentzian"):
        if type == "gaussian":
            return 1 - amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
        elif type == "lorentzian":
            return 1 - amp * (1 / (1 + ((x - cen) / sigma)**2))
        elif type == "voigt":
            raise NotImplementedError
        else:
            raise ValueError("Unknown shape type")

    @staticmethod
    def model(params, row):
        par_names = row["parameters"]
        amp = params[par_names["amp"]]
        cen = params[par_names["cen"]]
        sig = params[par_names["sig"]]
        x = row["clipped_wlc_to_velocity"]
        return __class__.shape(x, amp, cen, sig)


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

    def setup_parameters(self):
        "Step 4. Fit gaussian by using lmfit"
        self.params = lmfit.Parameters()
        "Initialise variables ..."
        l_parameter_names = []
        for _, row in self.df.iterrows():
            d = dict()
            amplitude = 'amp_line_{name}_epoch_{epoch}'.format(
                name=row["line_hash"], epoch=row["date"])

            self.params.add(amplitude, value=0.5,
                            vary=True)  # , min=0.01,max=1.0
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
            l_parameter_names.append(d)
        if "parameters" in self.df.columns:
            del self.df["parameters"]
        self.df["parameters"] = l_parameter_names
