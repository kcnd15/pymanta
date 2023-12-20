# manta_results.py


class MantaResults:
    """
    # Call:
    # manta(formula = biomarkers ~ ., data = patients)
    #
    # Type II Sum of Squares
    #
    #           Df Sum Sq Mean Sq F value      R2    Pr(>F)
    # age        1  400.6  400.63  7.3566 0.04242  0.001685 **
    # gender     1   34.3   34.28  0.6295 0.00363    0.5144
    # status     2 2152.7 1076.33 19.7643 0.22793 1.297e-12 ***
    # Residuals 91 4955.7   54.46
    # ---
    # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    # 4 observations deleted due to missingness
    """

    def __init__(self):
        self.result = ""
        self.residuals = None
        self.df_residuals = None
        self.ss_type = None

    def __repr__(self):
        return "Manta"

    def __str__(self):

        self.result += f"Type {self.ss_type} Sum of Squares\n\n"
        self.result += "Df Sum Sq Mean Sq F value      R2    Pr(>F)\n"
        self.result += f"\nResiduals: {self.df_residuals:.0f} {self.residuals:.1f}"
        self.result += "\n"

        return self.result

    def setResiduals(self, residuals):
        self.residuals = residuals

    def set_df_residuals(self, df_residuals):
        self.df_residuals = df_residuals

    def set_sumofsquares_type(self, ss_type):
        self.ss_type = ss_type
