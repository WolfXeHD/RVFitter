import unittest
import pkg_resources
import os
import matplotlib.pyplot as plt

from RVFitter import RVObject


def id_func(specsfile):
    filename = os.path.basename(specsfile)
    print('FILE NAME', specsfile)
    splitted_file = filename.split("_")
    starname = splitted_file[0]
    date = splitted_file[2]
    return starname, date


class TestRVObject(unittest.TestCase):
    test_datafile = pkg_resources.resource_filename(
        "RVFitter", "tests/test_data/B275_UVBVIS_20190605T07_barycor.nspec")
    line_list = pkg_resources.resource_filename(
        "RVFitter", "tests/test_data/spectral_lines_RVmeasurement.txt")
    line_list_debug = pkg_resources.resource_filename(
        "RVFitter", "tests/test_data/debug_spectral_lines_RVmeasurement.txt")

    mytest = RVObject.from_specsfile(starname="B275",
                                     date="20190605T07",
                                     specsfile=test_datafile,
                                     line_list=line_list)

    def test_plotting(self):
        line_list = pkg_resources.resource_filename(
            "RVFitter", "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        mytest = RVObject.from_specsfile_flexi(specsfile=self.test_datafile,
                                               id_func=id_func,
                                               line_list=line_list)
        for line in mytest.lines:
            fig, ax = plt.subplots()
            mytest.plot_line(line, ax=ax)
            figname = "DEBUG_" + line.line_name + ".png"
            plt.savefig(figname)
            #  plt.show()
            #  plt.close()

    def test_norming(self):
        test_datafile = pkg_resources.resource_filename(
            "RVFitter",
            "tests/test_data/B275_UVBVIS_20190605T07_barycor.nspec")
        line_list = pkg_resources.resource_filename(
            "RVFitter",
            "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        mytest = RVObject.from_specsfile(starname="B275",
                                         date="20190605T07",
                                         specsfile=test_datafile,
                                         line_list=line_list)

        limits = []
        limits.append([0.95, 1.01])
        limits.append([0.95, 1.01])
        limits.append([0.95, 1.01])
        for line, limit in zip(mytest.lines, limits):
            angstrom, flux, error = mytest.angstrom, mytest.flux, mytest.flux_errors
            line.add_normed_spectrum(angstrom=angstrom,
                                     flux=flux,
                                     error=error,
                                     leftValueNorm=limit[0],
                                     rightValueNorm=limit[1])
        for line in mytest.lines:
            fig, ax = plt.subplots()
            line.plot_normed_spectrum(ax)
            figname = "DEBUG_normed_" + line.line_name + ".png"
            plt.savefig(figname)

        for line in mytest.lines:
            left = line.line_profile - 5
            right = line.line_profile + 5
            line.clip_spectrum(leftClip=left, rightClip=right)



if __name__ == "__main__":
    unittest.main()
