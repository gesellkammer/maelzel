from math import pi, pow, sqrt


def computeShape(atten):
    """
    atten: sidelobe attenuation, in possitive dB
    """
    if atten < 0.:
        raise ValueError("Kaiser window shape must be computed from positive (> 0dB)"
            	         " sidelobe attenuation. (received attenuation < 0)")
    if atten > 60.0:
        alpha = 0.12438 * (atten + 6.3)
    elif atten > 13.26:
        alpha = ( 
            0.76609 * pow((atten - 13.26), 0.4) + 
            0.09834 * (atten - 13.26)
        )
    else:   
        # can't have less than 13dB.
        alpha = 0.0
    return alpha


def computeLength(width, sr, atten):
    """
    Returns the length in samples of a Kaiser window from the desired
    main lobe width.
    
    width: the width of the main lobe (Hz)
    sr: the sample rate, in samples / sec
    atten: the attenuation in possitive dB
    
    // ---------------------------------------------------------------------------
    //  computeLength
    // ---------------------------------------------------------------------------
    //  Compute the length (in samples) of the Kaiser window from the desired 
    //  (approximate) main lobe width and the control parameter. Of course, since 
    //  the window must be an integer number of samples in length, your actual 
    //  lobal mileage may vary. This equation appears in Kaiser and Schafer 1980
    //  (on the use of the I0 window class for spectral analysis) as Equation 9.
    //
    //  The main width of the main lobe must be normalized by the sample rate,
    //  that is, it is a fraction of the sample rate.
    //
    """
    normWidth = width / sr
    alpha = computeShape(atten)
    # the last 0.5 is cheap rounding. But I think I don't need cheap rounding 
    # because the equation from Kaiser and Schafer has a +1 that appears to be 
    # a cheap ceiling function.
    return int(1.0 + (2. * sqrt((pi*pi) + (alpha*alpha)) / (pi*normWidth)))