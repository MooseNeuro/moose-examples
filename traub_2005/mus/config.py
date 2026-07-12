# config.py ---
#
# Filename: config.py
# Description:
# Author: Subhasis Ray
# Created: Wed May 13 17:09:25 2026 (+0530)
#

# Code:
import logging

logger = logging.getLogger('traub2005')
# Add a logging handler to print messages to stderr
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(filename)s %(funcName)s: %(message)s')
ch.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)



#
# config.py ends here
