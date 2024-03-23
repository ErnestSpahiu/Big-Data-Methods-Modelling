using LinearAlgebra, JuMP, Ipopt, DelimitedFiles

"""
    a, b = hardSvm(X, y)

Computes the hard-SVM separating hyperplane for linearly separable classification data, using IPOPT in JuMP to solve a convex QP.

Written by Kamil Khan on March 16, 2024, for CHEMENG 4H03

# Inputs

- `X`: a matrix of training data points. The i^th row of `X` is the i^th point.
- `y`: a vector of labels for the training data. Each `y[i]` must be either `-1.0` or `1.0`.

# Outputs

- `a` and `b`: separating hyperplane coefficients. If hard-SVM was successful, each training data point `(x,y)` will satisfy `y*(dot(a, x) - b) > 0.0`.
"""
function hardSvm(X, y)
    nPoints = length(y)  # number of training points
    dimX = size(X, 2)    # dimension of each training point

    # initialize the optimization problem in JuMP
    problem = Model(Ipopt.Optimizer)

    # set up decision variables, which the optimization problem aims to choose
    @variable(problem, a[1:dimX])
    @variable(problem, b)

    # set up an objective function, which the optimization problem aims to minimize
    @objective(problem, Min, dot(a, a))

    # set up constraints, which the optimization problem's solution must satisfy
    @constraint(problem, hardThreshold[i=1:nPoints],
                y[i]*(dot(a, X[i,:]) - b) >= 1.0)

    # solve the optimization problem in JuMP
    optimize!(problem)

    # convert solution from JuMP objects to vectors/scalars
    aStar = value.(a)
    bStar = value.(b)

    return aStar, bStar
end

using DelimitedFiles
#test code
separable_data = readdlm("separable.csv", ',')
display(separable_data)
#X = first column
x = separable_data[:,1]
#y = second column
y = separable_data[:,2]

a, b = hardSvm(x, y)
println("a = ", a)
println("b = ", b)
#plot the data points and the separating hyperplane
using Plots
scatter(x, y, label="data points")
plot!(x -> (a[1]*x + b)/(-a[2]), label="separating hyperplane")
