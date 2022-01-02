# Flux balance analysis (FBA) - A mathematical simulation and modelling method

This project was centered on using flux-balance analysis to calculate the amount of lycopene in E.coli after inserting DNA for lycopene production. Using the iJE660 E.coli model of the COBRApy framework (https://github.com/opencobra/cobrapy) we now add the lycopene pathway by including three new metabolites

To see changes in the flux an therefore the production of lycopene we need to modify the pathway of E.coli in such a way that it produces more lycopene. This can be done by overexpressing genes or knocking out genes that hinder lycopene from being produced. We can use both methods as well as a combination of those to optimize the production of lycopene.
