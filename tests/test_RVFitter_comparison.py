import os
import unittest

from RVFitter.RVFitter import RVFitter
from RVFitter.RVFitter_comparison import RVFitter_comparison


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
        self.specsfilelist = os.path.join(os.path.dirname(__file__),
                                          'test_data/debug_specfile_list.txt')
        filename = os.path.join(os.path.dirname(self.specsfilelist),
                                "B275_speclist.pkl")
        self.myfitter = RVFitter.load_from_df_file(filename=filename)

    def test_table(self):
        collected_fitters = []
        for shape_profile in ["gaussian", "lorentzian"]:
            this_fitter = self.myfitter.fit_without_constraints(
                shape_profile=shape_profile)
            this_fitter.label = shape_profile + " without constraints"
            collected_fitters.append(this_fitter)

            this_fitter = self.myfitter.fit_with_constraints(
                shape_profile=shape_profile)
            this_fitter.label = shape_profile + " with constraints"
            collected_fitters.append(this_fitter)

        #  print(collected_fitterk

        comparer = RVFitter_comparison(list_of_fitters=collected_fitters)
        comparer.create_overview_df()
        print(comparer.df)


if __name__ == "__main__":
    unittest.main()
