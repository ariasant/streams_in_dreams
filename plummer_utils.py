import math


def enclosed_mass(r, M=1, b=1):
    return M*math.pow(r,3) / math.pow(math.pow(r,2) + math.pow(b,2), 3/2)

def crossing_time(r):
    """
    The time it would take to a particle to cross the radius at the
    circular velocity
    """
    M = enclosed_mass(r)
    v_circ = math.sqrt(M/r)
    return r / v_circ

def free_fall_time(r):
    """
    The free-fall time from radius r, computed from the mean density
    enclosed within r.
    """
    M = enclosed_mass(r)
    rho = M / (4/3 * math.pi * math.pow(r,3))
    return math.sqrt(3*math.pi / (32*rho))

def orbital_time(r):
    """
    The period of a circular orbit at radius r.
    """
    M = enclosed_mass(r)
    v_circ = math.sqrt(M/r)
    return 2*math.pi*r / v_circ

def density(r, M=1, b=1):
    return (3*M) / (4*math.pi*math.pow(b,3)) * math.pow(1 + math.pow(r,2)/math.pow(b,2), -5/2)
