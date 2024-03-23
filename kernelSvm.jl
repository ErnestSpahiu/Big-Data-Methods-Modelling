using LinearAlgebra, JuMP, Ipopt

"""
    a = kernelSvm(G, y; lambda=1e6)

Computes a soft-SVM linear classifier with the "kernel trick", using IPOPT in JuMP to solve a convex QP.

Written by Kamil Khan on March 20, 2024, for CHEMENG 4H03

# Inputs

- `G`: the Gram matrix of kernel evaluations for each pair of training inputs
- `y`: a vector of labels for the training data. Each `y[i]` must be either `-1.0` or `1.0`.
- `lambda=1e6`: the strictness/hardness of the soft threshold. Preferably large, and must be `> 0.0`

# Outputs

- `a`: classifier coefficients. Future inputs `x` are to be classified based on the sign of `sum(a[i]*K(xTrain[i]),x) for i = 1:n)`, where `xTrain[i]` was the i^th training input.
"""
function kernelSvm(G, y; lambda=1e6)
    nPoints = length(y)  # number of training points

    # initialize the optimization problem in JuMP
    problem = Model(Ipopt.Optimizer)

    # set up decision variables, which the optimization problem aims to choose
    @variable(problem, a[1:nPoints])
    @variable(problem, xi[1:nPoints] >= 0.0)

    # set up an objective function, which the optimization problem aims to minimize
    @objective(problem, Min, dot(a, G, a) + lambda*sum(xi))

    # set up constraints, which the optimization problem's solution must satisfy
    @constraint(problem, softThreshold[i=1:nPoints],
                y[i]*dot(G[i,:], a) >= 1.0 - xi[i])

    # solve the optimization problem in JuMP
    optimize!(problem)

    # convert solutions from JuMP objects to vectors/scalars
    aStar = value.(a)

    return aStar
end
