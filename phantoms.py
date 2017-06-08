"""Creating phantoms"""


import numpy as np 


def donut(discr, smooth=True, taper=20.0):
    """Return a 'donut' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 2:
        if smooth:
            return _donut_2d_smooth(discr, taper)
        else:
            return _donut_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _donut_2d_smooth(discr, taper):
    """Return a 2d smooth 'donut' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle_1(x):
        """Blurred characteristic function of an circle.

        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.26, 0.26)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.26, 0.26]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_circle_2(x):
        """Blurred characteristic function of an circle.

        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.115, 0.115)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.115, 0.115]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle_1) - discr.element(blurred_circle_2)
    return out.ufuncs.minimum(1, out=out)


def _donut_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'donut' phantom."""

    def circle_1(x):
        """Characteristic function of an ellipse.

        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.2, 0.2)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def circle_2(x):
        """Characteristic function of an circle.

        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.1, 0.1)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.1, 0.1]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle_1) - discr.element(circle_2)
    return out.ufuncs.minimum(1, out=out)
