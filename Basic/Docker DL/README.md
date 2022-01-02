# The CoLoMoTo Interactive Notebook

This small project was centered on the implementation and simulation of a given set of genes as a boolean regulatory network to analyze the quorum sensing of Vibrio fischeri.
To perform the simulations used the Jupyter Notebook in addition to the Docker Machine(https://colomoto.github.io/colomoto-docker/). Docker provides a variety of modelling and analyzing tool kits such as bioLQM, GINsim and Pint, which were run using python commands. Running those yields an enumeration of all possible states the system can be in as well as the system’s behaviour when set to a specific initial state.

To visualize the results we used GINsim which puts the model in the GINML format for displaying the layout information of the regulatory graphs and networks. We can visualize the activation and inhibition reactions between the nodes of the network, with nodes representing a gene and directed edges representing modulation of one node by another.
