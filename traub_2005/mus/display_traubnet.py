# display_traubnet.py ---
#
# Filename: display_traubnet.py
# Description:
# Author: Subhasis Ray
# Created: Sat Apr 26 10:04:02 2025 (+0530)
#

# Code:
from cortical_column import make_net, cell_counts, connection_spec
import vis
import adapter
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        model_root = '/model'
        populations = make_net(
            cell_counts=cell_counts,
            connection_spec=connection_spec,
            model_root=model_root,
        )
        graph = adapter.model_to_graph(model_root)
        print(graph)
        vis.display_network(graph)
    else:
        vis.display_data(sys.argv[1])

#
# display_traubnet.py ends here
