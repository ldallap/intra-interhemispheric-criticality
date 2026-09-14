# Intra- and Interhemispheric Signatures of Criticality at the Onset of Synchronization

Code associated with the paper:

**Intra- and Interhemispheric Signatures of Criticality at the Onset of Synchronization**

Leonardo Dalla Porta, Pietro Bozzo, Marco N. Pompili, Damien Depannemaecker, Antonio J. Fontenele, Tomoki Fukai, Pierpaolo Sorrentino, and Giovanni Rabuffo.

## Repository contents

This repository contains code for both the empirical data analysis and the computational model presented in the paper.

### Empirical analysis

The `empirical_analysis/` folder contains:

- `Empirical_DataAnalysis.ipynb` — analysis of the electrophysiological recordings.
- `Utils.py` — helper functions used by the notebook.

The notebook analyzes spiking activity from the left and right prefrontal cortex. It includes:

- spike raster visualization,
- pairwise spike-train correlations,
- intra- and interhemispheric correlations,
- neuronal avalanche detection,
- avalanche size and duration distributions,
- estimation of critical exponents,
- size-duration scaling,
- Distance to Criticality Coefficient (DCC).

The empirical analysis assumes that spike data are stored in a dictionary with the following structure:

```python
Spikes["Right_H"]["spiketime"]
Spikes["Right_H"]["neuronid"]

Spikes["Left_H"]["spiketime"]
Spikes["Left_H"]["neuronid"]
```

### Computational model

The `ComputationalModel/` folder contains the code used for the two-population active-rotator model.

Main files:

- `simulator.py` — simulation of two coupled neuronal populations.
- `analysis.py` — analysis functions for synchrony, susceptibility, neuronal avalanches, and criticality.
- `fig4.py` — reproduces the static model results shown in Figure 4.
- `Figure_story.py` — reproduces the model results describing slow fluctuations around criticality.
- `style.py` — common plotting settings.
- `custom_cmaps.py` — custom colormaps used for the figures.

The model includes local population dynamics, coupling between populations, shared input, and slow fluctuations in local excitability around the synchronization transition.

## Repository structure

```text
.
├── README.md
│
├── empirical_analysis/
│   ├── Empirical_DataAnalysis.ipynb
│   └── Utils.py
│
├── model/
│   ├── simulator.py
│   ├── analysis.py
│   └── plots/
│       ├── fig4.py
│       ├── Figure_story.py
│       ├── style.py
│       └── custom_cmaps.py
│
└── figures/
```

## Citation

If you use this code, please cite:

Dalla Porta L., Bozzo P., Pompili M. N., Depannemaecker D., Fontenele A. J., Fukai T., Sorrentino P., Rabuffo G.

**Intra- and Interhemispheric Signatures of Criticality at the Onset of Synchronization**  
https://www.biorxiv.org/content/10.64898/2025.12.11.693654v1
