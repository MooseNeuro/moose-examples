# test_spinystellate.py --- 
# 
# Filename: test_spinystellate.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Thu Apr 10 13:18:45 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 499
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 

# Code:

import unittest
from cell_test_util import SingleCellCurrentStepTest
import testutils
import cells
# 

simdt = 5e-6
plotdt = 0.25e-3
simtime = 1.0
    

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]



class TestSpinyStellate(SingleCellCurrentStepTest):
    def __init__(self, *args, **kwargs):
        self.celltype = 'SpinyStellate'
        SingleCellCurrentStepTest.__init__(self, *args, **kwargs)

    def setUp(self):
        SingleCellCurrentStepTest.setUp(self)

    def testVmSeriesPlot(self):
        self.runsim(simtime, pulse_list=self.pulse_list)
        self.plot_vm()

    def testChannelDensities(self):
        pass
        # equal = compare_cell_dump(self.dump_file, '../nrn/'+self.dump_file)
        # self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()



# 
# test_spinystellate.py ends here
