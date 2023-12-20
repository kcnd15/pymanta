# manta.py

from utils import scale, stop, match_arg
from manta_ss import manta_ss
from manta_results import MantaResults

import pandas as pd

from patsy.highlevel import dmatrix
import statsmodels.api as sm


def manta(
        biomarkers,  # array with biomarker values; R: formula,
        data,  # data, e.g. patients dataframe
        transform="none",  # "none", "sqrt", "log")
        ss_type="II",  # "I", "II", "III"
        contrasts=None,
        subset=None,
        fit=False
):
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
    # -> (96,5)
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

    # lmfit:
    # $coefficients
    #              biomarker1  biomarker2   biomarker3  biomarker4 biomarker5
    # (Intercept) -0.49025049 -1.19548036  0.124428353 -24.4062818 -0.3600481
    # age          0.00529589  0.01949775 -0.002838998   0.4094594  0.1334645
    # gender1      0.07256393  0.01608482  0.016514937  -0.4505664  0.3957010
    # status.L    -0.65077518 -0.53761407  0.017901769  -7.4342467 15.3203131
    # status.Q    -0.07236788 -0.14051561 -0.163548569  -6.0984142  9.9230951
    #
    # $residuals
    #      biomarker1   biomarker2   biomarker3  biomarker4   biomarker5
    # 1   -1.25803666  0.568447153 -0.128868984  -3.8029530   1.28453576
    # 2    0.35117254  0.009779514  0.020754049  -3.0447239  -1.73665335
    # 4   -3.33200972  0.859667419  0.374623224  -1.7538523   0.10277818
    # 5    0.88427874  0.016413759  0.368287825  -6.7873487  -0.80194012
    # 6    0.94011700  0.468583595  0.144586692   4.6233813  -2.81724456
    #
    # $fitted.values
    #      biomarker1  biomarker2  biomarker3 biomarker4 biomarker5
    # 1   -0.22304413 -0.06345397  0.09098338  2.7249687 -1.7843060
    # 2    0.26713962  0.07928171 -0.07475549 -2.4356566 -0.3400637
    # 4    0.24066017 -0.01820704 -0.06056050 -4.4829537 -1.0073860
    # 5   -0.05143683  0.06620441  0.10981826  3.8711330 -0.3255816
    # 6    0.00152207  0.26118190  0.08142828  7.9657273  1.0090630
    #
    # Call:
    # lm(formula = biomarkers ~ ., data = patients, contrasts = contr.list)
    #
    # Coefficients:
    #              biomarker1  biomarker2  biomarker3  biomarker4  biomarker5
    # (Intercept)   -0.490250   -1.195480    0.124428  -24.406282   -0.360048
    # age            0.005296    0.019498   -0.002839    0.409459    0.133464
    # gender1        0.072564    0.016085    0.016515   -0.450566    0.395701
    # status.L      -0.650775   -0.537614    0.017902   -7.434247   15.320313
    # status.Q      -0.072368   -0.140516   -0.163549   -6.098414    9.923095

    # fit linear regression model with statsmodels
    lmfit = sm.OLS(Y, X).fit()

    # ---------------------------------------------------------------------------------
    # Compute sums of squares, df's, pseudo-F statistics, partial R2s and eigenvalues
    # ---------------------------------------------------------------------------------

    #   stats <- manta2.ss(fit = lmfit, X = X, type = type, subset = subset)
    regressor_columns=mf_regressors.columns.to_list()
    stats = manta_ss(fit=lmfit, X=X, ss_type=ss_type, subset=subset, regressor_columns=regressor_columns)

    out = MantaResults()
    out.set_sumofsquares_type(ss_type)

    #   SS <- stats$SS
    #  SS
    #       age    gender    status Residuals
    #  400.6299   34.2810 2152.6636 4955.7163
    SS = stats["SS"]
    out.setResiduals(SS)

    #   df <- stats$df
    # df
    #       age    gender    status Residuals
    #         1         1         2        91
    out.set_df_residuals(stats["df_e"])

    #   f.tilde <- stats$f.tilde
    # f.tilde
    #        age     gender     status
    #  7.3566195  0.6294894 19.7642858

    #   r2 <- stats$r2
    # r2
    #         age      gender      status
    # 0.042419706 0.003629759 0.227929469

    #   e <- stats$e
    # e
    # [1] 38.66849176 13.05510597  2.00150461  0.66820365  0.06511482

    # ------------------
    # Compute P-values
    # ------------------
    #   l <- length(df) # SS[l], df[l] correspond to Residuals
    # -> 4L

    #   pv.acc <- mapply(p.asympt, ss = SS[-l], df = df[-l], MoreArgs = list(lambda = e))
    # pv.acc
    #               age       gender       status
    # [1,] 1.684575e-03 5.143919e-01 1.296629e-12
    # [2,] 1.000000e-14 1.000000e-14 1.000000e-14

    # -------------
    # ANOVA table
    # -------------
    # stats.l <- list(df, SS, SS/df, f.tilde, r2, pv.acc[1, ])
    #   cmat <- data.frame()
    #   for(i in seq(along = stats.l)) {
    #     for(j in names(stats.l[[i]])){
    #       cmat[j, i] <- stats.l[[i]][j]
    #     }
    #   }
    #   cmat <- as.matrix(cmat)
    # cmat
    #           V1        V2         V3         V4          V5           V6
    # age        1  400.6299  400.62988  7.3566195 0.042419706 1.684575e-03
    # gender     1   34.2810   34.28100  0.6294894 0.003629759 5.143919e-01
    # status     2 2152.6636 1076.33179 19.7642858 0.227929469 1.296629e-12
    # Residuals 91 4955.7163   54.45842         NA          NA           NA

    #   colnames(cmat) <- c("Df", "Sum Sq", "Mean Sq", "F value", "R2", "Pr(>F)")
    # cmat
    #           Df    Sum Sq    Mean Sq    F value          R2       Pr(>F)
    # age        1  400.6299  400.62988  7.3566195 0.042419706 1.684575e-03
    # gender     1   34.2810   34.28100  0.6294894 0.003629759 5.143919e-01
    # status     2 2152.6636 1076.33179 19.7642858 0.227929469 1.296629e-12
    # Residuals 91 4955.7163   54.45842         NA          NA           NA

    # --------
    # Output
    # --------
    # out <- list("call" = cl,
    #               "aov.tab" = cmat,
    #               "type" = type,
    #               "precision" = pv.acc[2, ],
    #               "transform" = transform,
    #               "na.omit" = lmfit$na.action)
    # out
    # $call
    # manta2(formula = biomarkers ~ ., data = patients)
    #
    # $aov.tab
    #           Df    Sum Sq    Mean Sq    F value          R2       Pr(>F)
    # age        1  400.6299  400.62988  7.3566195 0.042419706 1.684575e-03
    # gender     1   34.2810   34.28100  0.6294894 0.003629759 5.143919e-01
    # status     2 2152.6636 1076.33179 19.7642858 0.227929469 1.296629e-12
    # Residuals 91 4955.7163   54.45842         NA          NA           NA
    #
    # $type
    # [1] "II"
    #
    # $precision
    #    age gender status
    #  1e-14  1e-14  1e-14
    #
    # $transform
    # [1] "none"
    #
    # $na.omit
    #  3 47 64 96
    #  3 47 64 96
    # attr(,"class")
    # [1] "omit"

    #   if(fit){
    #     out$fit <- lmfit
    #   }
    # fit is FALSE

    #   ## Update class
    #   class(out) <- c('manta', class(out))
    #
    # out:
    #
    # Call:
    # manta2(formula = biomarkers ~ ., data = patients)
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

    #   return(out)

    return out
