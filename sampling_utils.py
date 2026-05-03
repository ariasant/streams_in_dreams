import numpy as np
from scipy.differentiate import derivative
from scipy.optimize import minimize_scalar
from scipy.integrate import quad



def sample_from_power_law_density_profile(
    n: int,
    r_min: float,
    r_max: float,
    alpha: int,
    rng = None,
    ):
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    if alpha <= 0:
        raise ValueError("alpha must be a positive integer")
    
    elif alpha == 3:
        
        inv_cdf = lambda x: r_min * np.power( r_max/r_min, x)

    else:
        
        # For other values of alpha, use the general inverse CDF
        A = np.power(r_max, 3-alpha) - np.power(r_min, 3-alpha)
        B = np.power(r_min, 3-alpha)
        
        inv_cdf = lambda x: np.power( A*x + B, (1/(3-alpha)) )

    # Sample numbers from a uniform distribution
    U = rng.uniform(size=n)
    
    return inv_cdf(U)


def make_isotropic_vectors(
    mag,
    rng = None
    ):
    
    
    try:
        unit = mag.unit
    except AttributeError:
        unit = 1.0
    
    if rng is None:
        rng = np.random.default_rng(42)

    theta = np.arccos(rng.uniform(-1, 1, size=len(mag)))
    phi = np.random.uniform(0, 2*np.pi, size=len(mag))

    x = mag * np.sin(theta) * np.cos(phi)
    y = mag * np.sin(theta) * np.sin(phi)
    z = mag * np.cos(theta)
    
    return np.array([x, y, z])*unit

# Density function from EXP
def rho(
    r,
    basis,
    coefs,
    t=None
    ):
    
    # Load basis coefficients
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))


    idx = basis.getFieldLabels().index("dens")
    
    r_flat = np.asarray(r).flatten()
    vals = []
    for xi in r_flat:
        vals.append(basis.getFields(xi, 0.0, 0.0)[idx])
        
    return np.array(vals).reshape(np.shape(r))
    
# Potential function EXP
def psi(
    r,
    basis,
    coefs,
    t=None):
    
    # Load basis coefficients
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))

    idx = basis.getFieldLabels().index("potl")
    
    r_flat = np.asarray(r).flatten()
    vals = []
    for xi in r_flat:
        vals.append(basis.getFields(xi, 0.0, 0.0)[idx])
        
    return np.array(vals).reshape(np.shape(r))
 
# Relative energy function 
def epsilon(
    r, 
    v, 
    basis,
    coefs,
    t=None,
    phi0=0.):

    # Compute the relative energy at radius r
    H = psi(r, basis, coefs, t) + 0.5*v*v
    
    return - H - phi0

# Functions to calculate the derivative of the density and potential fields
def drho_dr(
    r,
    basis,
    coefs,
    t=None):
    
    # Load basis coefficients
    
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))
    idx = basis.getFieldLabels().index("dens")
    
    
    def dens_fn(x):
        # 1. Flatten SciPy's multi-dimensional step arrays into a safe 1D list
        x_flat = np.asarray(x).flatten()
        vals = []
        
        for xi in x_flat:
            
            vals.append(basis.getFields(xi, 0.0, 0.0)[idx])
                
        # 4. Reshape the flat results back to the exact shape SciPy needs
        return np.array(vals).reshape(np.shape(x))
    
    
    derivative_results = derivative(dens_fn, 
                                    r,
                                    maxiter=100,
                                    initial_step=0.0001
                                    )

    
    return derivative_results.df


def dpsi_dr(
    r,
    basis,
    coefs,
    t=None):
    # This is just the radial force, which can be directly obtained from EXP
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))
    idx = basis.getFieldLabels().index("rad force")
    
    r_flat = np.asarray(r).flatten()
    vals = []
    for xi in r_flat:
        vals.append(- basis.getFields(xi, 0.0, 0.0)[idx])
        
    return np.array(vals).reshape(np.shape(r))
    
# The second derivative of the density and potential fields are computed numerically
def drho2_dr2(r,
    basis,
    coefs,
    t=None):
    
    # Load basis coefficients
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))
    idx = basis.getFieldLabels().index('dens')
    
    def get_field_derivative(r):
    
        def dens_fn(x):
            # 1. Flatten SciPy's multi-dimensional step arrays into a safe 1D list
            x_flat = np.asarray(x).flatten()
            vals = []
            
            for xi in x_flat:
                
                vals.append(basis.getFields(xi, 0.0, 0.0)[idx])
                    
            # 4. Reshape the flat results back to the exact shape SciPy needs
            return np.array(vals).reshape(np.shape(x))
    
    
        derivative_results = derivative(dens_fn, 
                                        r,
                                        maxiter=100,
                                        initial_step=0.0001
                                        )
        
        return derivative_results.df
    
    
    derivative_results = derivative(get_field_derivative,
                                    r,
                                    maxiter=100,
                                    initial_step=0.0001
                                    )
    
    return derivative_results.df


def dpsi2_dr2(
    r,
    basis,
    coefs,
    t=None):
        
    # Load basis coefficients
    if t is None:
        basis.set_coefs(coefs.getCoefStruct(coefs.Times()[-1]))
    else:
        basis.set_coefs(coefs.getCoefStruct(t))
    idx = basis.getFieldLabels().index("rad force")
    
    
    def dens_fn(x):
        # 1. Flatten SciPy's multi-dimensional step arrays into a safe 1D list
        x_flat = np.asarray(x).flatten()
        vals = []
        
        for xi in x_flat:
            
            vals.append(- basis.getFields(xi, 0.0, 0.0)[idx])
                
        # 4. Reshape the flat results back to the exact shape SciPy needs
        return np.array(vals).reshape(np.shape(x))
    
    
    derivative_results = derivative(dens_fn, 
                                    r,
                                    maxiter=100,
                                    initial_step=0.0001
                                    )

    
    return derivative_results.df


def drho2_dpsi2(
    r,
    basis,
    coefs,
    t=None):
    
    rho_prime = drho_dr(r, basis, coefs, t)
    psi_prime = dpsi_dr(r, basis, coefs, t)
    rho_double_prime = drho2_dr2(r, basis, coefs, t)
    psi_double_prime = dpsi2_dr2(r, basis, coefs, t)
    
    numerator = rho_double_prime * psi_prime - rho_prime * psi_double_prime
    denominator = psi_prime**3 + 1e-12
    
    return numerator / denominator


def evaluate_df(
    r,
    v_mag,
    second_derivative_spline
    ):
    
    eps = epsilon(r, v_mag)
    
    def integrand(u):
        psi_val = eps - u**2
        return second_derivative_spline(psi_val)
    
    # Perform the integration using numerical integration
    integral_val, _ = quad(integrand, 0, np.sqrt(eps))
        
    return - integral_val / (np.sqrt(2) * np.pi**2)




class VelocitySampler:
    """
    A class to pre-compute the target PDF and efficiently sample velocities 
    using vectorized rejection sampling.
    """
    def __init__(self, r, v_esc, rng=None):
        self.r = r
        self.v_esc = v_esc
        self.df_function = np.vectorize(evaluate_df)
        
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Calculate the maximum of the PDF *before* any sampling occurs
        self.max_pdf = self._compute_max_pdf()

    def _target_pdf(self, v):
        """The unnormalized probability density: v^2 * f(E)"""
        # Ensure v doesn't exceed v_esc to prevent unphysical values
        v = np.clip(v, 0, self.v_esc)
        return (v**2) * self.df_function(self.r, v)

    def _compute_max_pdf(self):
            
        # minimize_scalar looks for a minimum, so we pass the negative PDF
        res = minimize_scalar(
            lambda v: -self._target_pdf(v), 
            bounds=(0, self.v_esc), 
            method='bounded'
        )
        
        # Return the positive maximum, adding a 1% buffer for floating point safety
        return -res.fun * 1.01

    def sample(self, n_particles=1):
        """
        Draws n_particles using vectorized rejection sampling.
        """

        accepted_samples = []
        
        # Loop until we have accepted the requested number of particles
        while len(accepted_samples) < n_particles:
            
            # How many more particles do we need?
            n_needed = n_particles - len(accepted_samples)
            
            # Step A & B: Vectorized proposal of velocities and uniform heights
            # We draw a few extra to account for rejected samples
            v_prop = self.rng.uniform(0, self.v_esc, size=int(n_needed * 1.5))
            u = self.rng.uniform(0, 1, size=len(v_prop))
            
            # Step C: Evaluate the target PDF for all proposed velocities at once
            pdf_values = self._target_pdf(v_prop) / self.max_pdf
            
            # Boolean array of which samples fell under the curve
            accepted_mask = u <= pdf_values
            
            # Filter and append the accepted velocities
            valid_v = v_prop[accepted_mask]
            accepted_samples.extend(valid_v)

        # Return exactly the requested number of samples as a numpy array
        return np.array(accepted_samples[:n_particles])