# manta.py

from utils import scale


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

    # ------------------------------------------------
    # Save call, build model frame, obtain responses
    # ------------------------------------------------

    # --------
    # Checks
    # --------

    # TODO: remove NAs in biomarkers

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


    # response: num[1:96, 1:5] biomarker column 1:  -1.707, 0.392 -3.317 etc
    response = biomarkers

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

    # --------
    # Fit lm
    # --------

    # ---------------------------------------------------------------------------------
    # Compute sums of squares, df's, pseudo-F statistics, partial R2s and eigenvalues
    # ---------------------------------------------------------------------------------

    pass
