# LELEC2796_Project
Project for the course [LELEC2796](https://uclouvain.be/en-cours-2023-lelec2796) (2023-2024) <br>
By [Côme Wallner](https://github.com/elCarac) & [Jérôme Lafontaine](https://github.com/JeromeLafontaine)

**Teachers**: [Pr. Claude Oestges](https://en.wikipedia.org/wiki/Claude_Oestges) & [Pr. Luc Vandendorpe](https://uclouvain.be/fr/repertoires/luc.vandendorpe) <br>
**Teaching assistant**: [Jérome Eertmans](https://github.com/jeertmans)

---
This document will give informations about how to work with the files from this Github and how to get results.


## Introduction
The GitHub is organized as followed:
- 4 folders: **Matrices**, **Moving_fast**, **Moving_very_fast** and **results**
- 2 Jupyter notebooks: **project** and **project_channel_calculation** 
- 4 utils python codes: **utils_Alamouti**, **utils_eigen_TX2_RX1**, **utils_eigen** and **utils_TX2_RX1**


The main code to run is the **project.ipynb**. This notebook is based on **sionna** package from Python and the notebook will install it directly. The other python's packages used in this project are:


1. pandas
2. matplotlib
3. numpy
4. scipy
5. os
6. tensorflow

If one of these packages is not installed on your laptop, you can do the following command:

```bash
pip install <package>
```

## Documents' description
### project.ipynd description
This notebook will be based on the 4 utils codes and the **matrices** folder, more precisely the **channel_matrix** python code. The **channel_matrix** Python code will provide the different matrices used for the channel in the notebook. 

You just have to run all cells to obtain the results from this notebook. In the first slides the parameters can be chosen (the positions of the receivers (starting point and end points) have to be chosen with values linked to the position as explained in the code). The possible positions are represented in the article linked to this project.
#### 1. utils_Alamouty.py description
Files which contain the useful functions of the MIMO Alamouti coding.

#### 2. utils_eigen_TX2_RX1.py description
Files which contain the useful functions the the MISO dominant eigenmode transmission.


#### 3. utils_eigen.py description
Files which contain the useful functions the the MIMO dominant eigenmode transmission.

#### 4. utils_TX2_RX1.py description
Files which contain the useful functions of the MISO Alamouti coding.

### project_channel_calculation description
This is used to generate the channel matrix with ray tracing. The positions are hard coded and the code needs to be changed manually to change the positions of the UE (user equipment). 

