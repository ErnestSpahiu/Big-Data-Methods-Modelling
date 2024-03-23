using LinearAlgebra, JuMP, Ipopt

"""
    a, b = logisticRegression(X, y)

Uses logistic regression to determine a linear classifier, using IPOPT in JuMP to solve the training optimization problem.

Written by Kamil Khan on March 16, 2024, for CHEMENG 4H03

# Inputs

- `X`: a matrix of training data points. The i^th row of `X` is the i^th point.
- `y`: a vector of labels for the training data. Each `y[i]` must be either `0.0` or `1.0`.

# Outputs

- `a` and `b`: classifier's hyperplane coefficients. Ideally, many `y=1.0` training points should satisfy `dot(a, x) > b`, and many `y=0.0` points should satisfy `dot(a, x) < b`.
"""
function logisticRegression(X, y)
    nPoints = length(y)  # number of training points
    dimX = size(X, 2)    # dimension of each training point

    # initialize the optimization problem in JuMP
    problem = Model(Ipopt.Optimizer)

    # set up decision variables, which the optimization problem aims to choose
    @variable(problem, a[1:dimX])
    @variable(problem, b)

    # set up an objective function, which the optimization problem aims to minimize
    @objective(problem, Min, sum(log(1.0 + exp(-y[i]*(dot(a, X[i,:]) - b)))
                                 for i in 1:nPoints))
    
    # solve the optimization problem in JuMP
    optimize!(problem)

    # convert solution from JuMP objects to vectors/scalars
    aStar = value.(a)
    bStar = value.(b)

    return aStar, bStar
end
