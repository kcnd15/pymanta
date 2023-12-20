# manta_ss.py

import numpy as np
from utils import crossproduct


def manta_ss(
        fit,
        x,
        ss_type="II",
        subset=None,
        tol=1e-3
    ):
    """
    MANTA sum of squares
    :param fit: fitted linear regression model
    :param x: predictors
    :param ss_type: sum of squares type
    :param subset:
    :param tol:
    :return:
    """

    # fit:
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
    #
    # X:
    #     (Intercept) age gender1      status.L   status.Q
    # 1             1  53      -1 -7.850462e-17 -0.8164966
    # 2             1  48       1 -7.071068e-01  0.4082483
    # 4             1  43       1 -7.071068e-01  0.4082483
    # 5             1  58       1 -7.850462e-17 -0.8164966
    # 6             1  68       1 -7.850462e-17 -0.8164966

    # ----------------------------------------------------------
    # Residual sum-of-squares and cross-products (SSCP) matrix
    # ----------------------------------------------------------
    # fit$residuals: (96,5)
    # lmfit$residuals
    #      biomarker1   biomarker2   biomarker3  biomarker4   biomarker5
    # 1   -1.25803666  0.568447153 -0.128868984  -3.8029530   1.28453576
    # 2    0.35117254  0.009779514  0.020754049  -3.0447239  -1.73665335
    # 4   -3.33200972  0.859667419  0.374623224  -1.7538523   0.10277818
    # 5    0.88427874  0.016413759  0.368287825  -6.7873487  -0.80194012
    # 6    0.94011700  0.468583595  0.144586692   4.6233813  -2.81724456

    fit_residuals = fit.resid

    # SSCP.e <- crossprod(fit$residuals)
    # SSCP.e
    #              biomarker1 biomarker2  biomarker3 biomarker4   biomarker5
    # biomarker1  189.7426567 -20.176633   0.5968432 -151.72961  -52.8805334
    # biomarker2  -20.1766327  65.486963   3.5840320  -49.28826   25.0580754
    # biomarker3    0.5968432   3.584032   6.2346621  -19.84405    0.2067679
    # biomarker4 -151.7296104 -49.288261 -19.8440518 3472.66720 -299.2267528
    # biomarker5  -52.8805334  25.058075   0.2067679 -299.22675 1221.5848102

    SSCP_e = crossproduct(fit_residuals) # fit_residuals.transpose().dot(fit_residuals)

    # --------------------------------
    # Residual sum-of-squares and df
    # --------------------------------
    # SS.e <- sum(diag(SSCP.e)) -> 4955.716
    SS_e = np.sum(SSCP_e.diagonal())

    # df.e <- fit$df.residual # df.e <- (n-1) - sum(df) -> 91
    df_e = fit.df_resid

    # ----------------------
    # Total sum-of-squares
    # ----------------------
    # SS.t <- sum(diag(crossprod(fit$model[[1L]]))) -> 9444.429
    # fit$model[[1L]]:
    #       biomarker1   biomarker2   biomarker3   biomarker4 biomarker5
    # 1   -1.481080791  0.504993185 -0.037885609  -1.07798430 -0.4997702
    # 2    0.618312159  0.089061220 -0.054001445  -5.48038044 -2.0767170
    # 4   -3.091349542  0.841460380  0.314062719  -6.23680605 -0.9046078
    # 5    0.832841917  0.082618173  0.478106085  -2.91621569 -1.1275217
    # 6    0.941639068  0.729765500  0.226014973  12.58910858 -1.8081816
    fit_model = fit.model.endog
    SS_t = np.sum(crossproduct(fit_model).diagonal())

    # -------------------------
    # Partial sums-of-squares
    # -------------------------
    # terms <- attr(fit$terms, "term.labels") # Model terms
    # terms
    # [1] "age"    "gender" "status"

    # n.terms <- length(terms)
    # n.terms: 3

    # if(!is.null(subset)) {
    #   if(all(subset %in% terms)) {
    #     iterms <- which(terms %in% subset)
    #   } else {
    #    stop(sprintf("Unknown terms in subset: %s",
    #    paste0("'", subset[which(! subset %in% terms)], "'",
    #    collapse = ", ")))
    #   }
    #   } else {
    #   iterms <- 1:n.terms
    #   -> iterms 1 2 3
    # }
    # asgn <- fit$assign
    # asgn: 0 1 2 3 3

    # df <- SS <- numeric(n.terms) # Initialize empty
    # -> df, SS: 0 0 0

    # names(df) <- names(SS) <- terms
    # -> names(df), names(SS): "age", "gender", "status"

    # if (type == "I"){
    #
    # effects <- as.matrix(fit$effects)[seq_along(asgn), , drop = FALSE]
    #
    # for (i in iterms) {
    #  subs <- which(asgn == i)
    #  SS[i] <- sum(diag(crossprod(effects[subs, , drop = FALSE])))
    #  df[i] <- length(subs)
    # }
    #
    # } else {
    #   sscp <- function(L, B, V){
    #   LB <- L %*% B
    #   crossprod(LB, solve(L %*% tcrossprod(V, L), LB))
    # }

    # B <- fit$coefficients     # Coefficients
    # V <- solve(crossprod(X))  # V = (X'X)^{-1}
    # p <- nrow(B)
    # I.p <- diag(p)

    # In contrast to car::Anova, intercept
    # information is not returned for
    # type III sums-of-squares

    # if (type == "III"){
    #
    # for (i in iterms){
    #   subs <- which(asgn == i)
    #   L <- I.p[subs, , drop = FALSE] # Hypothesis matrix
    #   SS[i] <- sum(diag(sscp(L, B, V)))
    #   df[i] <- length(subs)
    # }
    #
    # } else {
    #
    # is.relative <- function(term1, term2, factors) {
    #   all( !( factors[, term1] & ( !factors[, term2] ) ) )
    # }

    # fac <- attr(fit$terms, "factors")
    # for (i in iterms) {
    #   term <- terms[i]
    #   subs.term <- which(asgn == i)
    #   if(n.terms > 1) { # Obtain relatives
    #   relatives <- (1:n.terms)[-i][sapply(terms[-i],
    #                                    function(term2)
    #                                    is.relative(term, term2, fac))]
    # } else {
    #    relatives <- NULL
    # }
    # subs.relatives <- NULL
    # for (relative in relatives) {
    #    subs.relatives <- c(subs.relatives, which(asgn == relative))
    # }
    # L1 <- I.p[subs.relatives, , drop = FALSE] # Hyp. matrix (relatives)
    # if (length(subs.relatives) == 0) {
    #  SSCP1 <- 0
    #  } else {
    #    SSCP1 <- sscp(L1, B, V)
    # }
    # L2 <- I.p[c(subs.relatives, subs.term), , drop = FALSE] # Hyp. matrix (relatives + term)
    # SSCP2 <- sscp(L2, B, V)
    # SS[i] <- sum(diag(SSCP2 - SSCP1))
    # df[i] <- length(subs.term)
    # }
    # }
    # }

    # --------
    # subset
    # --------
    # if(!is.null(subset)){
    #  SS <- SS[iterms]
    #  df <- df[iterms]
    # }

    # SS
    #       age    gender    status
    #  400.6299   34.2810 2152.6636
    # SS.e: 4955.716, SS.t: 9444.429
    # df
    #    age gender status
    #      1      1      2
    # df.e: 91

    # ----------
    # pseudo-F
    # ----------
    # f.tilde <- SS/SS.e*df.e/df
    # f.tilde
    #        age     gender     status
    #  7.3566195  0.6294894 19.7642858

    # -----------
    # r.squared
    # -----------
    # R2 <- (SS.t - SS.e)/SS.t
    # SS.t: 9444.429, SS.e: 4955.716
    # R2: 0.4752762
    R2 = (SS_t - SS_e) / SS_t
    pass

    # R2adj <- 1-( (1-R2)*(n-1) / df.e )
    # n:

    # r2 <- SS/SS.t
    # r2
    #         age      gender      status
    # 0.042419706 0.003629759 0.227929469

    # r2adj <- 1-( (1-r2)*(n-1) / df.e )

    # Get eigenvalues from cov(R)*(n-1)/df.e
    # e <- eigen(SSCP.e/df.e, symmetric = T, only.values = T)$values
    # e
    # [1] 38.66849176 13.05510597  2.00150461  0.66820365  0.06511482

    # e <- e[e/sum(e) > tol]
    # e
    # [1] 38.66849176 13.05510597  2.00150461  0.66820365  0.06511482

    # ------------------
    # result structure
    # ------------------

    # return(list("SS" = c(SS, "Residuals" = SS.e),
    # "df" = c(df, "Residuals" = df.e),
    # "f.tilde" = f.tilde, "r2" = r2, "e" = e))

    result_dict = {
        "SS": SS_e,
        "df_e": df_e,
    }

    return result_dict
