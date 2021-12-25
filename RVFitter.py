import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import lmfit
import hashlib
import pandas as pd

# TODO: add a unique identifier for a line which can be used as key for parameters

c = 299792.458

class Line(object):
    def __init__(self, line_name, line_profile, wlc_window):
        self.line_name = line_name
        self.line_profile = line_profile
        self.wlc_window = wlc_window

        hash_object = hashlib.md5(self.line_name.encode())
        self.hash = hash_object.hexdigest()[:10]

        self.normed_wlc = None
        self.normed_flux = None
        self.normed_errors = None
        self.leftValueNorm = None
        self.rightValueNorm = None
        self.leftClip = None
        self.rightClip = None
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

    def add_normed_spectrum(self, angstrom, flux, error, leftValueNorm,
                            rightValueNorm):
        self.got_normed = True
        self.spline = splrep([min(angstrom), max(angstrom)],
                             [leftValueNorm, rightValueNorm],
                             k=1)
        self.continuum = splev(angstrom, self.spline)
        self.normed_wlc = angstrom
        self.normed_flux = flux / self.continuum
        self.normed_error = error
        self.leftValueNorm = leftValueNorm
        self.rightValueNorm = rightValueNorm

    def clip_spectrum(self, leftClip, rightClip):
        if not self.got_normed:
            print("You cannot clip before you norm!")
            raise SystemExit
        else:
            self.leftClip = leftClip
            self.rightClip = rightClip
            masking = (self.normed_wlc > self.leftClip) & (self.normed_wlc <
                                                           self.rightClip)
            self.clipped_wlc = self.normed_wlc[masking]
            self.clipped_flux = self.normed_flux[masking]

    def plot_normed_spectrum(self, ax=None):

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

        if self.normed_error is None:
            print("No errors for normed flux - assuming 1%")
            self.normed_error = 0.01 * self.normed_flux
        ax.errorbar(self.normed_wlc[indices],
                    self.normed_flux[indices],
                    yerr=self.normed_error[indices],
                    fmt='ro',
                    color='black',
                    ecolor='black')
        ax.axhline(y=1, linestyle='-.', color='black')
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_ylabel('flux')
        ax.set_title(self.line_name + ' ' + str("%.0f" % self.line_profile))


class RVFitter(lmfit.Model):
    """Docstring for RVFitter. """
    def __init__(self, rvobjects, line_list):
        self.rvobjects = rvobjects
        self.line_list = line_list
        self.line_names, self.line_profiles, self.wlc_windows = self._read_line_list(
            self.line_list)
        self.lines = self.make_line_objects()
        self.objective = None
        self.objective_set = False
        self.params = None
        self.star = list(set([item.starname for item in self.rvobjects]))
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

    def create_df(self):
        l_dfs = []
        for rvobject in self.rvobjects:
            df = rvobject.make_dataframe()
            l_dfs.append(df)
        self.df = pd.concat(l_dfs, axis=0)

    def set_objective(self, objective):
        self.objective_set = True
        self.objective = objective

    def sort_by_date(self):
        """
        sorts the rvobjects by their date
        """
        dates = [rvo.date for rvo in self.rvobjects]
        sorted_index = np.argsort(dates)
        self.rvobjects = [self.rvobjects[i] for i in sorted_index]

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
            self.params.add(cen, value=row["line_profile"],
                            vary=True)  # , min=0.01,max=1.0
            d["cen"] = cen
            sig = 'sig_line_{name}_epoch_{epoch}'.format(name=row["line_hash"],
                                                         epoch=row["date"])
            self.params.add(sig, value=0.3, vary=True)  # , min=0.01,max=1.0
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
        self.df["parameters"] = l_parameter_names

    def get_parameters(self, group):
        l_parameter = []
        for _, row in self.df.iterrows():
            l_parameter.append(row["parameters"][group])
        return l_parameter

    def constrain_parameters(self, group):
        parameters_to_constrain = self.get_parameters(group=group)
        for idx, par in enumerate(parameters_to_constrain):
            if idx == 0:
                par_to_constrain = par
                continue
            self.params[par].expr = par_to_constrain

    def print_fit_result(self):
        output_file = 'Params_' + self.star + '.dat'
        file = open(output_file, 'w')

        for _, row in self.df.iterrows():
            line_profile = row["line_profile"]
            #  vsini_constant = c / (2 * line_profile * pow(np.log(2), 0.5))

            #        print 'line 4 read= ',ix+1,' -- ',line_profile
            #        for iy, y in enumerate(fileContent): # iterate over number of epochs
            iy = 0
            amplitude, err_amplitude = round(self.result.params[row["parameters"]["amp"]].value, 4), round(
                self.result.params[row["parameters"]["amp"]].stderr, 4)
            sigma, err_sigma = round(self.result.params[row["parameters"]["sig"]].value, 4), round(
                self.result.params[row["parameters"]["sig"]].stderr, 4)
            fwhm, err_fwhm = round(self.result.params[row["parameters"]["fwhm"]].value, 4), round(
                self.result.params[row["parameters"]["fwhm"]].stderr, 4)
            centroid, err_centroid = round(self.result.params[row["parameters"]["cen"]].value, 4), round(
                self.result.params[row["parameters"]["cen"]].stderr, 4)
            #  vsini, err_vsini = round(vsini_constant * fwhm, 4), round(vsini_constant * err_fwhm, 4)
            height, err_height = round(0.3183099 * amplitude / max(1.e-15, sigma), 4), round(
                0.3183099 * err_amplitude / max(1.e-15, sigma), 4)
            print('-----------', row["line_name"] + ' ' + str(line_profile), '-----------')
            print('Amplitude= ', '\t', amplitude, ' +/-\t', err_amplitude)
            print('Height= ', '\t', height, ' +/-\t', err_height)
            #            print 'FWHM=     ','\t',fwhm,' +/-\t',err_fwhm
            print('Sigma=     ', '\t', sigma, ' +/-\t', err_sigma)
            print('Centroid=     ', '\t', centroid, ' +/-\t', err_centroid)
            print('RV=     ', '\t', c * ((centroid - line_profile) / line_profile), ' +/-\t',
                  (err_centroid / centroid) * c)
            #            print 'vsini=        ','\t',vsini,' +/-\t',err_vsini

            file.write('\n')
            file.write('Line profile ' + row["line_name"] + '\t' + str(line_profile) + '\n')
            #            file.write('Amplitude '+'\t'+str(amplitude)+'\t'+str(err_amplitude)+'\n')
            file.write('Amplitude ' + '\t' + str(height) + '\t' + str(err_height) + '\n')
            #            file.write('FWHM '+'\t'+str(fwhm)+'\t'+str(err_fwhm)+'\n')
            file.write('sigma ' + '\t' + str(sigma) + '\t' + str(err_sigma) + '\n')
            file.write('centroid ' + '\t' + str(centroid) + '\t' + str(err_centroid) + '\n')
            file.write('RV_line ' + '\t' + str(((centroid - line_profile) / line_profile) * 299792.458) + '\t' + str(
                (err_centroid / centroid) * 299792.458) + '\n')
        #           file.write('vsini '+'\t'+str(vsini)+'\t'+str(err_vsini)+'\n')

        file.close()

    def run_fit(self):
        if self.objective_set is False:
            print("You did not specify an objective function!")
            raise SystemExit
        self.result = lmfit.minimize(self.objective,
                                     self.params,
                                     args=([self.df]))

    @classmethod
    def from_specsfilelist_flexi(cls,
                                 specsfilelist,
                                 id_func,
                                 line_list,
                                 datetime_formatter="%Y%m%dT%H"):
        rvobjects = [
            RVObject.from_specsfile_flexi(
                specsfile=specsfile,
                id_func=id_func,
                datetime_formatter=datetime_formatter,
                line_list=line_list) for specsfile in specsfilelist
        ]
        return cls(rvobjects=rvobjects, line_list=line_list)


class RVObject(object):
    """Docstring for RVObject. """
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

    def get_line_by_hash(self, hash):
        for line in self.lines:
            if line.hash == hash:
                return line
        else:
            print("Line with name {name} is unknown.".format(name=name))
            print("Known lines are:")
            for line in self.lines:
                print("\t", line.line_name)

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
        """classmethod for creation of RVObject
        :arg1: starname - name of the star
        :arg2: date - date to be formatted into datetime object
        :arg3: specsfile - path to specsfile which should be read

        :returns: RVObject
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
                             datetime_formatter="%Y%m%dT%H"):
        """classmethod for creation of RVObject
        :arg1: specsfile - path to specsfile which should be read
        :arg2: id_func - function which processes the specsfile string and returns starname and date

        :returns: RVObject
        """
        starname, date = id_func(specsfile)

        wavelength, flux, flux_errors = cls._read_specsfile(specsfile)
        lines = cls.make_line_objects(line_list)

        return cls(starname=starname,
                   date=date,
                   wavelength=wavelength,
                   flux=flux,
                   flux_errors=flux_errors,
                   lines=lines,
                   datetime_formatter=datetime_formatter)

    def plot_line(self, line, clipping=False, ax=None):
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
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_ylabel('flux')
        ax.set_title(line_name + ' ' + str("%.0f" % cl))

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

    def get_line_from_row(self):
        pass

    def make_dataframe(self):
        df = pd.DataFrame()
        dict_for_df = dict()
        dict_for_df["star_name"] = self.starname
        dict_for_df["date"] = self.date
        dict_for_df["wavelength"] = [np.array(self.wavelength)]
        dict_for_df["angstrom"] = [np.array(self.angstrom)]
        dict_for_df["flux"] = [np.array(self.flux)]
        dict_for_df["flux_error"] = [self.flux_errors]
        dict_for_df["date_time_obj"] = self.date_time_obj
        for line in self.lines:
            dict_for_df["line_name"] = line.line_name
            dict_for_df["line_profile"] = line.line_profile
            dict_for_df["wlc_window"] = line.wlc_window
            dict_for_df["normed_wlc"] = [np.array(line.normed_wlc)]
            dict_for_df["normed_flux"] = [np.array(line.normed_flux)]
            dict_for_df["normed_error"] = [np.array(line.normed_error)]
            dict_for_df["leftValueNorm"] = line.leftValueNorm
            dict_for_df["rightValueNorm"] = line.rightValueNorm
            dict_for_df["leftClip"] = line.leftClip
            dict_for_df["rightClip"] = line.rightClip
            dict_for_df["clipped_wlc"] = [np.array(line.clipped_wlc)]
            dict_for_df["clipped_flux"] = [np.array(line.clipped_flux)]
            dict_for_df["clipped_error"] = [np.array(line.clipped_error)]
            dict_for_df["line_hash"] = line.hash

            this_df = pd.DataFrame.from_dict(dict_for_df)
            df = df.append(this_df)
        return df
