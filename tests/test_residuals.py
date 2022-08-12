import unittest
import os
import pkg_resources
from RVFitter import RVFitter


class TestRVFitter(unittest.TestCase):
    def setUp(self):
        #      self.line_list = pkg_resources.resource_filename(
        #          "RVFitter",
        #          "tests/test_data/debug_spectral_lines_RVmeasurement.txt")
        self.specsfilelist = os.path.join(os.path.dirname(__file__),
                                          'test_data/debug_specfile_list.txt')

    #
    #      tmp_specsfilelist = get_tmp_file(self.specsfilelist)
    #
    #      self.myfitter = RVFitter.from_specsfilelist_name_flexi(
    #          specsfilelist_name=tmp_specsfilelist,
    #          id_func=id_func,
    #          line_list=self.line_list,
    #          share_line_hashes=True
    #          )


if __name__ == "__main__":
    unittest.main()
