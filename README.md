# LELEC2796_Project
Project for the course [LELEC2796](https://uclouvain.be/en-cours-2023-lelec2796) (2023-2024) <br>
By [Côme Wallner](https://github.com/elCarac) & [Jérôme Lafontaine](https://github.com/JeromeLafontaine)

**Teachers**: [Pr. Claude Oestges](https://en.wikipedia.org/wiki/Claude_Oestges) & [Pr. Luc Vandendorpe](https://uclouvain.be/fr/repertoires/luc.vandendorpe) <br>
**Teaching assistant**: [Jérome Eertmans](https://github.com/jeertmans)

---
In this document, informations about how to work with the files from this Github will be given and how to get results.


## Introduction
The github is organized as followed:
- 4 folders: **Matrices**, **Moving_fast**, **Moving_very_fast** and **results**
- 2 Jupyter notebooks: **project** and **project_channel_calculation** 
- 4 utils python codes: **utils_Alamouti**, **utils_eigen_TX2_RX1**, **utils_eigen** and **utils_TX2_RX1**


The main code to run is the **project.ipynb**. This notebook is based on **sionna** package from python and the notebook will install it directly. The other python's packages used in this project are:


1. pandas
2. matplotlib
3. numpy
4. scipy
5. os
6. tensorflow

If one of these packages are not installed on your laptop, you can do the following command:

```bash
pip install <package>
```

## Documents' description
### project.ipynd description
This notebook will be based on the 4 utils codes and the **matrices** folder, more precisely the **channel_matrix** python code. The **channel_matrix** python code will provide the different matrices used for the channel in the notebook.

To obtain the results from this notebook, you just have to run all cells.
#### 1. utils_Alamouty.py description

#### 2. utils_eigen_TX2_RX1.py description

#### 3. utils_eigen.py description

#### 4. utils_TX2_RX1.py description

### project_channel_calculation description

