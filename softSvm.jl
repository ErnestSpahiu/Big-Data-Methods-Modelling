using LinearAlgebra, JuMP, Ipopt

"""
    a, b = softSvm(X, y; lambda=1e6)

Computes a soft-SVM linear classifier, using IPOPT in JuMP to solve a convex QP.

Written by Kamil Khan on March 16, 2024, for CHEMENG 4H03

# Inputs

- `X`: a matrix of training data points. The i^th row of `X` is the i^th point.
- `y`: a vector of labels for the training data. Each `y[i]` must be either `-1.0` or `1.0`.
- `lambda=1e6`: the strictness/hardness of the soft threshold. Preferably large, and must be `> 0.0`

# Outputs

- `a` and `b`: classifier's hyperplane coefficients. Ideally, many training data points `(x,y)` should satisfy `y*(dot(a, x) - b) > 0.0`.
"""
function softSvm(X, y; lambda=1e6)
    nPoints = length(y)  # number of training points
    dimX = size(X, 2)    # dimension of each training point

    # initialize the optimization problem in JuMP
    problem = Model(Ipopt.Optimizer)

    # set up decision variables, which the optimization problem aims to choose
    @variable(problem, a[1:dimX])
    @variable(problem, b)
    @variable(problem, xi[1:nPoints] >= 0.0)

    # set up an objective function, which the optimization problem aims to minimize
    @objective(problem, Min, dot(a, a) + lambda*sum(xi))

    # set up constraints, which the optimization problem's solution must satisfy
    @constraint(problem, softThreshold[i=1:nPoints],
                y[i]*(dot(a, X[i,:]) - b) >= 1.0 - xi[i])

    # solve the optimization problem in JuMP
    optimize!(problem)

    # convert solutions from JuMP objects to vectors/scalars
    aStar = value.(a)
    bStar = value.(b)

    return aStar, bStar
end
