# manta.py

from utils import scale, stop, match_arg
import numpy as np
import pandas as pd


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


    # mf[[1L]] <- Y

    # ------------------
    # Define contrasts
    # ------------------

    # --------------------
    # Build model matrix
    # --------------------
    # X <- model.matrix(mt, mf, contr.list)

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

    # ---------------------------------------------------------------------------------
    # Compute sums of squares, df's, pseudo-F statistics, partial R2s and eigenvalues
    # ---------------------------------------------------------------------------------

    #   stats <- manta2.ss(fit = lmfit, X = X, type = type, subset = subset)
    #   SS <- stats$SS
    #   df <- stats$df
    #   f.tilde <- stats$f.tilde
    #   r2 <- stats$r2
    #   e <- stats$e

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
