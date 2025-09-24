# Traub 2005 model in MOOSE
## Redesigned by Subhasis Ray in 2025

The original moose implementation of the Single Column Thalamocortical Model by Traub et al, 2005 used a lot of metaprogramming with metafix. While this was clever, the code turned out to be hard to read and maintain. In the mean time, MOOSE has also changed making some things simpler to implement.

This directory contains a reimplementation of the Traub et al 2005 model using modern Python (Python 3) and MOOSE > 4.1.0.
