import casadi as cs
import numpy as np
import scipy.interpolate as spi

class bspline_curve:
    def __init__(self, control_points):
        """
        Initialize a B-spline curve object with the given control points.

        Parameters
        ----------
        `control_points`: `np.ndarray` of shape `(n, 2)`
        """
        self.control_points = control_points
        self.tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)
        self.knots = self.tck[0]
        self.coeffs = self.tck[1]
        self.degree = self.tck[2]

        t = cs.MX.sym('t')
        coeffs_matrix = cs.horzcat(*self.coeffs).T
        curve = cs.bspline(t, coeffs_matrix, [self.knots.tolist()], [self.degree], 2, {})
        self.curve_func = cs.Function('curve_func', [t], [curve])
        tangent = cs.jacobian(curve, t)
        tangent /= cs.norm_2(tangent)
        normal = cs.vertcat(-tangent[1], tangent[0])
        self.tangent_func = cs.Function('tangent_func', [t], [tangent])
        self.normal_func = cs.Function('normal_func', [t], [normal])

        self.lim_surf_A = np.diag([1.0, 1.0, self.get_curvature()])

        self.t_samples = np.linspace(0, 1, 10000)
        self.pt_samples = self.sample_pts(self.t_samples)
        self.psic_samples = np.arctan2(self.pt_samples[:, 1], self.pt_samples[:, 0])
        permute_idx = np.argsort(self.psic_samples)
        self.t_samples = self.t_samples[permute_idx]
        self.psic_samples = self.psic_samples[permute_idx]
        deduplicate_idx = np.where(np.diff(self.psic_samples) > 0)[0]
        self.t_samples = self.t_samples[deduplicate_idx]
        self.psic_samples = self.psic_samples[deduplicate_idx]
        tck_psic2t = spi.splrep(self.psic_samples, self.t_samples, s=0.1, per=True)
        psic = cs.MX.sym('psic')
        psic2t = cs.bspline(psic, cs.horzcat(*tck_psic2t[1]).T, [tck_psic2t[0].tolist()], [tck_psic2t[2]], 1, {})
        self.psic_to_t_func = cs.Function('psic_to_t_func', [psic], [psic2t])

    def psic_to_t(self, psic):
        """
        Convert the azimuth angle to the parameter of the B-spline curve.

        Parameters
        ----------
        `psic`: `float`
            The azimuth angle. Unit: rad

        Returns
        -------
        `float`
            The parameter of the B-spline curve.
        """
        # psic = cs.fmod(psic, 2 * cs.pi)
        # psic = cs.if_else(cs.le(psic, 0), psic + 2 * cs.pi, psic)
        # return psic / (2 * cs.pi)
        return self.psic_to_t_func(psic)
    
    def integrate(self, f, N=1000, M=1000):
        """
        Integrate the given function in the area enclosed by the B-spline curve, using Green's theorem.

        Parameters
        ----------
        `f`: `function`
            The integrand function, such as:
            ```
            def f(x, y):
                return np.sqrt(x ** 2 + y ** 2)
            ```

        Returns
        -------
        `float`
            The integral value.
        """
        def t_to_xy(t):
            pts = np.array(spi.splev(t, self.tck))
            return pts[0, :], pts[1, :]
        
        def t_to_dxdy(t):
            pts = np.array(spi.splev(t, self.tck, der=1))
            return pts[0, :], pts[1, :]
        
        t_samples = np.linspace(0, 1, N)
        x_samples, y_samples = t_to_xy(t_samples)
        dx_samples, dy_samples = t_to_dxdy(t_samples)

        s_samples = np.linspace(0, 1, M)
        t_grid, s_grid = np.meshgrid(t_samples, s_samples, indexing='ij')

        x_t = x_samples.reshape(-1, 1)
        y_t = y_samples.reshape(-1, 1)
        
        x_points = s_grid * x_t
        y_points = s_grid * y_t
        
        f_values = f(x_points, y_points)
        jacobian = s_grid * (x_t * dy_samples.reshape(-1, 1) - y_t * dx_samples.reshape(-1, 1))
        integrand = f_values * jacobian
        integral = np.trapz(np.trapz(integrand, s_samples, axis=1), t_samples)
        
        return integral
    
    def get_curvature(self):
        """
        Get the curvature squared of the B-spline curve.

        Returns
        -------
        `float`
            The curvature squared value.
        """
        area = self.integrate(lambda x, y: 1)
        integral = self.integrate(lambda x, y: np.sqrt(x ** 2 + y ** 2))
        c = integral / area
        return 1.0 / (c ** 2)
    
    def sample_pts(self, t_samples):
        pt_samples = np.array(spi.splev(t_samples, self.tck)).T

        return pt_samples
    
if __name__ == "__main__":
    import time

    control_points = np.array([[-0.0878332 ,  0.00894283],
                               [-0.06927735, -0.00348606],
                               [-0.05296399, -0.01158518],
                               [-0.04714031, -0.01441735],
                               [-0.041185  , -0.01692681],
                               [-0.0356633 , -0.02305051],
                               [-0.02967586, -0.02596084],
                               [-0.02266738, -0.02777073],
                               [-0.00996478, -0.02773122],
                               [-0.00041986, -0.02406837],
                               [ 0.0044366 , -0.02229121],
                               [ 0.0168373 , -0.02315491],
                               [ 0.03243288, -0.02825957],
                               [ 0.0445449 , -0.02781803],
                               [ 0.06018997, -0.02223057],
                               [ 0.06313941, -0.01095734],
                               [ 0.063856  ,  0.00187829],
                               [ 0.06498363,  0.00934821],
                               [ 0.06430348,  0.02594651],
                               [ 0.06331416,  0.0364197 ],
                               [ 0.04723743,  0.04102006],
                               [ 0.03250469,  0.04007967],
                               [ 0.0155726 ,  0.03512453],
                               [ 0.00858525,  0.03455442],
                               [ 0.00335342,  0.035297  ],
                               [-0.00586487,  0.0390531 ],
                               [-0.01101431,  0.04052551],
                               [-0.02407005,  0.04040656],
                               [-0.02948088,  0.03901881],
                               [-0.03477284,  0.03622134],
                               [-0.04096227,  0.03262363],
                               [-0.04923069,  0.0288909 ],
                               [-0.06533611,  0.02522149],
                               [-0.07393795,  0.0175605 ],
                               [-0.0878332 ,  0.00894283]])
    curve = bspline_curve(control_points)
    
    N = 10000

    t_samples = np.linspace(0, 1, N)

    start_time = time.time()
    for i in range(N):
        curve.curve_func(t_samples[i])
    print(f"Time consumed for calculate curve point for {N} times: {time.time() - start_time}")

    start_time = time.time()
    spi.splev(t_samples, curve.tck)
    print(f"Time consumed for numerically calculate curve point for {N} times: {time.time() - start_time}")

    start_time = time.time()
    for i in range(N):
        curve.tangent_func(t_samples[i])
    print(f"Time consumed for calculate tangent for {N} times: {time.time() - start_time}")

    start_time = time.time()
    for i in range(N):
        curve.normal_func(t_samples[i])
    print(f"Time consumed for calculate normal for {N} times: {time.time() - start_time}")

    start_time = time.time()
    for i in range(N):
        tangent = curve.tangent_func(t_samples[i])
        normal = np.array([-tangent[1], tangent[0]])
    print(f"Time consumed for calculate tangent and normal for {N} times: {time.time() - start_time}")

    psic_samples = np.linspace(-np.pi, np.pi, N)
    start_time = time.time()
    for i in range(N):
        curve.psic_to_t_func(psic_samples[i])
    print(f"Time consumed for calculate psic2t for {N} times: {time.time() - start_time}")

    start_time = time.time()
    for i in range(N):
        spi.splev(t_samples[i], curve.tck, der=1)
    print(f"Time consumed for numerically calculate tangent for {N} times: {time.time() - start_time}")

    start_time = time.time()
    for i in range(N):
        tangent = np.array(spi.splev(t_samples[i], curve.tck, der=1))
        norm = np.array([-tangent[1], tangent[0]])
    print(f"Time consumed for numerically calculate tangent and normal for {N} times: {time.time() - start_time}")