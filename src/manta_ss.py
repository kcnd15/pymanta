# manta_ss.py

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

    pass

    ## Residual sum-of-squares and cross-products (SSCP) matrix
    # SSCP.e <- crossprod(fit$residuals)

    ## Residual sum-of-squares and df
    # SS.e <- sum(diag(SSCP.e))
    # df.e <- fit$df.residual # df.e <- (n-1) - sum(df)

    ## Total sum-of-squares
    # SS.t <- sum(diag(crossprod(fit$model[[1L]])))

    ## Partial sums-of-squares
    # terms <- attr(fit$terms, "term.labels") # Model terms
    # n.terms <- length(terms)

    # if(!is.null(subset)) {
    #   if(all(subset %in% terms)) {
    #     iterms <- which(terms %in% subset)
    #   }else {
    #    stop(sprintf("Unknown terms in subset: %s",
    #    paste0("'", subset[which(! subset %in% terms)], "'",
    #    collapse = ", ")))
    #   }
    #   } else {
    #   iterms <- 1:n.terms
    # }
    # asgn <- fit$assign

    # df <- SS <- numeric(n.terms) # Initialize empty
    # names(df) <- names(SS) <- terms

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
    #
    # sscp <- function(L, B, V){
    # LB <- L %*% B
    # crossprod(LB, solve(L %*% tcrossprod(V, L), LB))
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
    # all( !( factors[, term1] & ( !factors[, term2] ) ) )
    # }

    # fac <- attr(fit$terms, "factors")
    # for (i in iterms){
    # term <- terms[i]
    # subs.term <- which(asgn == i)
    # if(n.terms > 1) { # Obtain relatives
    # relatives <- (1:n.terms)[-i][sapply(terms[-i],
    #                                    function(term2)
    #                                    is.relative(term, term2, fac))]
    # } else {
    #    relatives <- NULL
    # }
    # subs.relatives <- NULL
    # for (relative in relatives){
    #    subs.relatives <- c(subs.relatives, which(asgn == relative))
    # }
    # L1 <- I.p[subs.relatives, , drop = FALSE] # Hyp. matrix (relatives)
    # if (length(subs.relatives) == 0) {
    #  SSCP1 <- 0
    #  } else {
    # SSCP1 <- sscp(L1, B, V)
    # }
    # L2 <- I.p[c(subs.relatives, subs.term), , drop = FALSE] # Hyp. matrix (relatives + term)
    # SSCP2 <- sscp(L2, B, V)
    # SS[i] <- sum(diag(SSCP2 - SSCP1))
    # df[i] <- length(subs.term)
    # }
    # }
    # }

    ## subset
    # if(!is.null(subset)){
    #  SS <- SS[iterms]
    #  df <- df[iterms]
    # }

    ## pseudo-F
    # f.tilde <- SS/SS.e*df.e/df

    ## r.squared
    # R2 <- (SS.t - SS.e)/SS.t
    # R2adj <- 1-( (1-R2)*(n-1) / df.e )
    # r2 <- SS/SS.t
    # r2adj <- 1-( (1-r2)*(n-1) / df.e )

    # Get eigenvalues from cov(R)*(n-1)/df.e
    # e <- eigen(SSCP.e/df.e, symmetric = T, only.values = T)$values
    # e <- e[e/sum(e) > tol]

    # return(list("SS" = c(SS, "Residuals" = SS.e),
    # "df" = c(df, "Residuals" = df.e),
    # "f.tilde" = f.tilde, "r2" = r2, "e" = e))

    return None
