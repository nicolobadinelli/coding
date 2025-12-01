# Cosmic Birefringence & Axion Dynamics #

This folder contains three Jupyter notebooks forming a coherent numerical pipeline to study the background evolution of an axion-like field.
The notebooks are meant to be read in order, as each step builds upon the previous one.

# Notebook Structure #

1. 1-Background_evolution.ipynb

Goal:
Compute and analyze the background dynamics of an axion-like field $\chi$ driving (early) dark energy or anisotropic cosmic birefringence.

Contents:
	•	Equation of motion for the homogeneous scalar field
	•	Dependence on the axion mass $m_\chi$
	•	Evolution of field, equation of state, and density parameter
	•	Figures reproducing the dynamics used later to compare with CLASS

Output / Importance:
This notebook gives an accurate approximate description of the backgroud evolution of the field $\chi$


2. 2-Evolution_with_CLASS.ipynb

Goal:
Implement the axion model into CLASS, solving for background evolution 

Contents:
	•	Overview of the CLASS modifications
	•	Implementation of the scalar potential
	•	Comparison between CLASS and numerical expectations

Output / Importance:
This notebook produces more accurate evolution of the background field, its equation of state and density parameter


3. 3-Consistency_Check.ipynb

Goal:
Verify the correctness of the CLASS modifications and of the numerical results.

Contents:
	•	Consistency checks between background evolution from CLASS and the standalone notebook

Output / Importance:
Explains why the two previous notebook were not exactly identical to each other, even if the two approaches are correct


# How to Use This Folder

	1.	Start with Notebook 1 to understand the physics and produce reference plots.
	2.	Move to Notebook 2 to run the CLASS-based computation.
	3.	Use Notebook 3 to check what's wrong with the numerical calculations.


# Requirements

    •	Python ≥ 3.9
    •	CLASS
    •	Standard scientific libraries: numpy, scipy, matplotlib, h5py
    •	Jupyter Notebook / JupyterLab


# Notes
	•	The notebooks are intended to be part of the workflow of Nicolò’s Thesis project.
	•	Each notebook is self-contained but conceptually linked to the others.
	•	Plots, equations, and comments follow the theoretical framework described in:
		    Greco, Bartolo & Gruppuso (2023) — Probing Axions through Tomography of Anisotropic Cosmic Birefringence (https://arxiv.org/abs/2211.06380).
