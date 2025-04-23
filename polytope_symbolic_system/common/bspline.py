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