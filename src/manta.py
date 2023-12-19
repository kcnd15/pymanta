# manta.py

from utils import scale, stop, match_arg
from manta_ss import manta_ss

import numpy as np
import pandas as pd

from patsy.highlevel import dmatrix
from sklearn import linear_model


def manta(
        biomarkers,  # array with biomarker values; R: formula,
        data,  # data, e.g. patients dataframe
        transform="none",  # "none", "sqrt", "log")
        ss_type="II",  # "I", "II", "III"
        contrasts=None,
        subset=None,
        fit=False
):
    print("function manta")

    # --------
    # Checks
    # --------
    # transform <- match.arg(transform, c("none", "sqrt", "log"))
    # type <- match.arg(type, c("I", "II", "III"))
    transform = match_arg(transform, ["none", "sqrt", "log"], argument_name="transform")
    ss_type = match_arg(ss_type, ["I", "II", "III"], argument_name="ss_type")

    # ------------------------------------------------
    # Save call, build model frame, obtain responses
    # ------------------------------------------------
    #   cl <- match.call() -> manta(formula = biomarkers ~ ., data = patients)
    #   m <- match(c("formula", "data"), names(cl), 0L) -> match returns a vector of the positions of
    #        (first) matches of its first argument in its second.
    #   mf <- cl[c(1L, m)] -> manta2(formula = biomarkers ~ ., data = patients)
    #   mf$drop.unused.levels <- TRUE -> manta2(formula = biomarkers ~ ., data = patients, drop.unused.levels = TRUE)
    #   mf$na.action = "na.omit" -> manta2(formula = biomarkers ~ ., data = patients, drop.unused.levels = TRUE,
    #     na.action = "na.omit")

    # The rows with at least one NA either in Y or X
    # (only considering variables used in the formula)
    # will be removed before transforming / centering
    #  mf[[1L]] <- quote(stats::model.frame) -> stats::model.frame(formula = biomarkers ~ ., data = patients,
    #     drop.unused.levels = TRUE, na.action = "na.omit")
    #   mf <- eval(mf, parent.frame())
    #   -> new dataframe combining biomarker and patient data:
    #   biomarkers.biomarker1 biomarkers.biomarker2 biomarkers.biomarker3 biomarkers.biomarker4 biomarkers.biomarker5 age gender  status
    # 1 -1.707049e+00          2.075822e+00          1.879238e-01          2.923038e+01          1.588166e+00  53   male    mild
    # 2 3.923442e-01          1.659890e+00          1.718080e-01          2.482799e+01          1.121899e-02  48 female healthy
    # 4 -3.317318e+00          2.412289e+00          5.398722e-01          2.407156e+01          1.183328e+00  43 female healthy
    #   mt <- attr(mf, "terms") -> biomarkers ~ age + gender + status, and more lines
    #   response <- model.response(mf, "numeric")
    # -> response variables of the model, i.e. biomarkers:
    #       biomarker1 biomarker2   biomarker3 biomarker4   biomarker5
    # 1   -1.707048753 2.07582190 1.879238e-01   29.23038 1.588166e+00
    # 2    0.392344197 1.65988994 1.718080e-01   24.82799 1.121899e-02
    # 4   -3.317317504 2.41228910 5.398722e-01   24.07156 1.183328e+00
    # 5    0.606873955 1.65344689 7.039155e-01   27.39215 9.604143e-01
    # 6    0.715671106 2.30059422 4.518244e-01   42.89748 2.797545e-01

    # combine response biomarkers and patient predictors into a single dataframe
    mf = pd.concat([biomarkers, data], axis=1)

    # remove NaN
    mf = mf.dropna(axis="index")

    # regressors dataframe containing age, gender, status
    mf_regressors = mf.iloc[:, 5:]
    mf_regressors.reset_index(inplace=True, drop=True)

    # convert Pandas dataframe biomarkers (first 5 columns) to a numpy array
    mf_biomarkers = mf.iloc[:, :5]
    response = mf_biomarkers.to_numpy()

    # --------
    # Checks
    # --------
    #   if (NCOL(response) < 2){
    #     -> NCOL(response) = 5, columns of biomarkers
    #     stop("The number of response variables should be >= 2")
    #   }
    #   if (length(attr(mt, "term.labels")) < 1) {
    #     -> here 3, age, gender and status
    #     stop("The model should contain at least one predictor (excluding the intercept)")
    #   }


    # ----------------------------------------------------
    # Transform and center responses, update model frame
    # ----------------------------------------------------

    # if(transform == "none"){
    #   Y <- response
    # } else if (transform == "sqrt"){
    #   if (any(response < 0)) {
    #      stop("'sqrt' transformation requires all response values >= 0")
    #   }
    #   Y <- sqrt(response)
    # } else if (transform == "log"){
    # if (any(response <= 0)) {
    #    stop("'log' transformation requires all response values > 0")
    # }
    # Y <- log(response)
    # }

    Y = None
    if transform == "none":
        Y = response

    # Y <- scale(Y, center = TRUE, scale = FALSE)
    # sample Y values: num [1:96, 1:5] biomarker1: -1.481 0.618 -3.091
    # scale is generic function whose default method centers and/or scales the columns of a numeric matrix.
    #
    # Usage
    # scale(x, center = TRUE, scale = TRUE)
    Y = scale(Y, center=True, scale_flag=False)

    # replace the response matrix with the scaled response matrix
    #     biomarkers.biomarker1 biomarkers.biomarker2 biomarkers.biomarker3 biomarkers.biomarker4 biomarkers.biomarker5 age gender  status
    # 1            -1.481080791           0.504993185          -0.037885609          -1.077984302          -0.499770229  53   male    mild
    # 2             0.618312159           0.089061220          -0.054001445          -5.480380440          -2.076717042  48 female healthy
    # 4            -3.091349542           0.841460380           0.314062719          -6.236806048          -0.904607816  43 female healthy
    # 5             0.832841917           0.082618173           0.478106085          -2.916215688          -1.127521745  58 female    mild
    # 6             0.941639068           0.729765500           0.226014973          12.589108580          -1.808181583  68 female    mild
    # mf[[1L]] <- Y

    Y_df = pd.DataFrame(Y)

    mf = pd.concat([Y_df, mf_regressors], axis=1)

    # ------------------
    # Define contrasts
    # ------------------
    #     if(is.null(contrasts)){
    #     contrasts <- list(unordered = "contr.sum", ordered = "contr.poly")
    #     -> $unordered [1] "contr.sum", $ordered [1] "contr.poly"
    #     dc <- attr(mt, "dataClasses")[-1]
    #     -> dc
    #           age        gender    status
    #           "numeric"  "factor"  "ordered"
    #     contr.list <- lapply(dc, FUN = function(k){
    #       # No contrast for quantitative predictors
    #       # Sum contrasts for unordered categorical predictors
    #       # Polynomial contrasts for ordered categorical predictors
    #       contr.type <- switch(k, "factor" = contrasts$unordered,
    #                            "ordered" = contrasts$ordered)
    #       return(contr.type)
    #     })
    #     -> $age NULL, $gender [1] "contr.sum", $status [1] "contr.poly"
    #     contr.list <- contr.list[!unlist(lapply(contr.list, is.null))]
    #     -> $gender [1] "contr.sum", $status [1] "contr.poly"
    #   } else {
    #     contr.list <- contrasts
    #   }

    if contrasts is None:
        contrasts = None

    # --------------------
    # Build model matrix
    # --------------------
    # X <- model.matrix(mt, mf, contr.list)
    # ->
    #     (Intercept) age gender1      status.L   status.Q
    # 1             1  53      -1 -7.850462e-17 -0.8164966
    # 2             1  48       1 -7.071068e-01  0.4082483
    # 4             1  43       1 -7.071068e-01  0.4082483
    # 5             1  58       1 -7.850462e-17 -0.8164966

    age = mf["age"].to_list()
    gender = mf["gender"].to_list()
    status = mf["status"].to_list()
    X = dmatrix("age + gender + status")

    # --------
    # Fit lm
    # --------

    #   lmfit <- lm.fit(X, Y)
    #   class(lmfit) <- c("mlm", "lm")
    #   lmfit$na.action <- attr(mf, "na.action")
    #   lmfit$contrasts <- attr(X, "contrasts")
    #   lmfit$xlevels <- .getXlevels(mt, mf)
    #   lmfit$call <- cl[c(1L, m)]
    #   lmfit$call[[1L]] <- quote(lm)
    #   if(length(contr.list) > 0) lmfit$call$contrasts <- quote(contr.list)
    #   lmfit$terms <- mt
    #   lmfit$model <- mf

    # Create linear regression object
    lmfit = linear_model.LinearRegression()

    # Train the model using the training sets
    lmfit.fit(X, Y)

    # ---------------------------------------------------------------------------------
    # Compute sums of squares, df's, pseudo-F statistics, partial R2s and eigenvalues
    # ---------------------------------------------------------------------------------

    #   stats <- manta2.ss(fit = lmfit, X = X, type = type, subset = subset)
    #   SS <- stats$SS
    #   df <- stats$df
    #   f.tilde <- stats$f.tilde
    #   r2 <- stats$r2
    #   e <- stats$e

    stats = manta_ss(fit=lmfit, x=X, ss_type=ss_type, subset=subset)

    # ------------------
    # Compute P-values
    # ------------------
    #   l <- length(df) # SS[l], df[l] correspond to Residuals
    #   pv.acc <- mapply(p.asympt, ss = SS[-l], df = df[-l], MoreArgs = list(lambda = e))

    # -------------
    # ANOVA table
    # -------------

    # --------
    # Output
    # --------

    pass
