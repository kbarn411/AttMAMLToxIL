## Directory with the dataset

created in 2026 by Karol Baran
karol.baran[at]pg.edu.pl

## Instructions

Data on ionic liquids toxicity as presented in https://doi.org/10.1021/acs.estlett.3c00106 by Yan et al. were utilized. To train the models, one should first download or prepare a file with data on ILs' toxicity named Data.xlsx in this directory. If one would like to run the scripts on their data, the file Data.xlsx should follow the structure of Yan's et al. file. To reproduce the experiments and retain comparability to the original Yan et al. work, it is advisable to remain consistent regarding data splits with the file from their ILToxDB GitHub repository (Data.xlsx file from https://github.com/YanLabAI/ILTox). The supporting data (subset B of the database) can be obtained after registration from https://www.iltox.com/. If one would like to simply use Yan et al. data as they are (subset A), please review their article and repository and then run: 

`wget https://github.com/YanLabAI/ILTox/blob/main/Data.xlsx`