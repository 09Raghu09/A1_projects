# Simulation/Spreading models

The code example is based on the kaggle notebook created by the user Lisphilar. It uses python
and a basic framework of libraries e.g pandas, sklearn, datetime. The main data used is from the World Health Organization (WHO) showing novel corona infections by country. Furthermore
supplementary data is used to include the age pyramid for each country. The WHO data set
is preprocessed to include the variables: Date, Country, Province, Confirmed, Infected, Deaths
and Recovered.


This project deals with the application of a SIR-model on current COVID-19 case
data taken either from a city (e.g. Berlin) or national (e.g. Germany) scale. The model itself is
extended beyond the simple case by integrating two new states (Exposed, Dead) to the model and by
studying the impact of independent features (ICUbed-capacity, Age, Smoking, and Gender) on the
epidemic. By fitting the model to actual case data, possible projections can be made. Furthermore,
different scenarios such as lockdown, reducing social contacts, and wearing masks, are explored
by simulating their effect on the fitted model. Each prevention method is simulated over different
periods and in combination with and without wearing masks on top.

This project is to use agent-based simulation to model the interactions of individuals
within a population during the COVID-19 outbreak, so that one can determine how small changes
in behavior and interaction can affect population level output. Different extensions (incubation
and exposed state; chronic conditions and comorbidities; central locations) are implemented to
refine the model. In the end, the variability of human behaviour can be shown with the purpose to
understand the variability in the likely effectiveness of proposed interventions.
