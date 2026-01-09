"""
new file containing gain class and functions
part of rewrite to fix gain implementation
plan to implement:

seperate gain from dispersion operator
modify rhs function to linearly add gain component to rv
modify main solving method to no longer use solve_ivp and instead perform rk45 step by step such that
    step size can be passed to rhs for correct calculation of gain"""