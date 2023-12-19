# usage.py

from data import get_biomarkers, get_patients

# library(manta)
# manta2(biomarkers ~ ., data = patients)
#
# Call:
# manta(formula = biomarkers ~ ., data = patients)
#
# Type II Sum of Squares
#
# Df Sum Sq Mean Sq F value      R2    Pr(>F)
# age        1  400.6  400.63  7.3566 0.04242  0.001685 **
# gender     1   34.3   34.28  0.6295 0.00363    0.5144
# status     2 2152.7 1076.33 19.7643 0.22793 1.297e-12 ***
# Residuals 91 4955.7   54.46
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 4 observations deleted due to missingness

# read toy test data
biomarkers = get_biomarkers()
patients = get_patients()

print("Patients:\n", patients.head())
print("Biomarkers:\n", biomarkers.shape)

