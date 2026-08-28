from jax.lax import fori_loop, pmean, while_loop
import jax.numpy as jnp


def cg_solve_fori_loop(A, b, x0=None, A_diag=None, tol=1e-5, atol=0, maxiter=1000):
    """
    Solve the system Ax = b using the conjugate gradient method with a preconditioner.
    Iterates for a fixed number of iterations, and tracks the solution with the smallest residual "on the fly".
    
    Parameters:
    - A: Function that applies the matrix-vector multiplication A * d.
    - b: Right-hand side vector of the system.
    - x0: Initial guess for the solution (optional).
    - A_diag: Diagonal elements of the preconditioner matrix.
    - tol: Relative tolerance for convergence (default: 1e-5).
    - atol: Absolute tolerance for convergence (default: 0).
    - maxiter: Fixed number of iterations (default: 1000).
    
    Returns:
    - x_best: The solution vector with the smallest residual.
    - r_best: The smallest residual.
    - i_best: The iteration number corresponding to the best residual.
    """
    
    # Preconditioner: inverse of the diagonal elements of A
    M_inv = 1 / A_diag

    # Initialize x0 if not provided
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # Initial residual and search direction
    r_0 = b - A(x0)
    z_0 = M_inv * r_0
    p_0 = z_0

    # Initialize best solution, residual, and iteration
    r_best = jnp.sqrt(jnp.vdot(r_0, r_0))
    x_best = x0
    i_best = 0

    def iteration(i, val):
        """Perform a single CG iteration and update the best solution on the fly."""
        x, z, p, r, x_best, r_best, i_best = val

        Ap = A(p)
        alpha = jnp.vdot(r, z) / jnp.vdot(p, Ap)

        x_next = x + alpha * p
        r_next = r - alpha * Ap

        z_next = M_inv * r_next
        beta = jnp.vdot(r_next, z_next) / jnp.vdot(r, z)

        p_next = z_next + beta * p

        # Compute current residual norm
        r_norm = jnp.sqrt(jnp.vdot(r_next, r_next))

        # Update best solution, residual, and iteration if the current one is smaller
        x_best = jnp.where(r_norm < r_best, x_next, x_best)
        i_best = jnp.where(r_norm < r_best, i, i_best,)
        r_best = jnp.where(r_norm < r_best, r_norm, r_best) 

        return (x_next, z_next, p_next, r_next, x_best, r_best, i_best)

    # Initialize values for the loop
    init_val = (x0, z_0, p_0, r_0, x_best, r_best, i_best)

    # Run the fixed number of iterations using fori_loop
    _, _, _, r_last, x_best, r_best, i_best = fori_loop(0, maxiter, iteration, init_val)

    return x_best, (r_best, i_best) 


def cg_solve_while_loop(A, b, x0, A_diag, map_axis_name, tol=1e-5, atol=0, maxiter=1000):
    """
    Solve the system f = Sx using the conjugate gradient method.

    Parameters:
    - A: Function that applies the matrix-vector multiplication A * d.
    - b: Right-hand side vector of the system.
    - x0: Initial guess for the solution (vector).
    - A_diag: Diagonal elements of the preconditioner matrix S.
    - tol: Relative tolerance for convergence.
    - atol: Absolute tolerance for convergence.
    - maxiter: Maximum number of iterations.

    Returns:
    - dp_i: The solution vector.
    - residual: The final residual after convergence.
    """

    # Preconditioner: element-wise reciprocal of the diagonal
    conditioner = 1 / A_diag

    # Calculate squared norm of f_i for tolerance scaling
    b_norm_squared = jnp.vdot(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * b_norm_squared, jnp.square(atol))

    def convergence_check(val):
        """Check if convergence criteria are met."""
        i, _, r, _ = val  # Unpack iteration variables
        r_norm_squared = jnp.vdot(r, r)  # Compute squared residual norm
        r_norm_squared = pmean(r_norm_squared, axis_name=map_axis_name)
        return (i < maxiter) & (r_norm_squared > atol2)

    def iteration(val):
        """Perform a single CG iteration."""
        i, x, r, d = val  # Unpack iteration variables

        # Compute matrix-vector product Ad
        Ad = A(d)

        # Compute r^T M r
        rMr = jnp.vdot(r, conditioner * r)

        # Compute step size alpha
        alpha = rMr / jnp.vdot(Ad, d)

        # Update solution and residual vectors
        x_next = x + alpha * d
        r_next = r - alpha * Ad

        # Compute preconditioned residual and update search direction
        Md = conditioner * r_next
        beta = jnp.vdot(r_next, Md) / rMr
        d_next = Md + beta * d

        return (i + 1, x_next, r_next, d_next)

    # Initialize values
    if x0 is None:
        x0 = jnp.zeros_like(b)

    r0 = b - A(x0)  # Initial residual

    d0 = conditioner * r0  # Initial search direction

    init_val = (0, x0, r0, d0)  # Initial iteration values

    # Run the CG algorithm using a while loop
    niter, x_best, r_best, _ = while_loop(convergence_check, iteration, init_val)

    return x_best, (r_best, niter)
