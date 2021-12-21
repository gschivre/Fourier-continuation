using LinearAlgebra, FFTW, Polynomials

function finitediff_coeff(stencil, d)
    # adapted from finitedifference.jl
    n = length(stencil)
    A = [Rational{BigInt}(s ^ i) for i in 0:(n - 1), s in stencil]
    b = zeros(Rational{BigInt}, n)
    b[d + 1] = factorial(big(d))
    return Float64.(A \ b)
end

function forward_stencil(d, o)
    c = 2 * ((d + 1) >>> 1) - 1 + o
    if (d % 2) == 0
        return 0:c
    else
        return 0:(c - 1)
    end
end
backward_stencil(d, o) = -reverse(forward_stencil(d, o))

forward_fdm(d, o) = finitediff_coeff(forward_stencil(d, o), d)
backward_fdm(d, o) = finitediff_coeff(backward_stencil(d, o), d)

function boundary_data_matrix(f, d, o)
    n = length(f) - 1
    return [[f[1]; f[end]] [(n ^ j) * (i == 1 ? dot(forward_fdm(j, o), view(f, forward_stencil(j, o) .+ 1)) : dot(backward_fdm(j, o), view(f, backward_stencil(j, o) .+ (n + 1)))) for i in 1:2, j in 1:d]]
end

function P0(m, r)
    p = SparsePolynomial(1.0)
    for n in 1:(r - m)
        p += SparsePolynomial(Dict(n => ((-1.0) ^ n) * binomial(r + n, n)))
    end
    return p * SparsePolynomial(Dict(m => 1)) * fromroots(-ones(r + 1)) / factorial(m)
end

function P1(m, r)
    p = SparsePolynomial(1.0)
    for n in 1:(r - m)
        p += fromroots(-ones(n)) * binomial(r + n, n)
    end
    return p * fromroots(-ones(m)) * SparsePolynomial(Dict((r + 1) => (-1.0) ^ (r + 1))) / factorial(m)
end

function continuation_polynomial(F)
    r = size(F, 2) - 1
    p = F[1, 1] * P0(0, r) + F[2, 1] * P1(0, r)
    for m in 1:r
        p += F[1, m + 1] * P0(m, r) + F[2, m + 1] * P1(m, r)
    end
    return p
end

# extend the function from [0, 1] to [-1, 1] using 2 points Hermite interpolation with d the derivative order and o the approximation order
function ext_func(f, d, o) 
    P = continuation_polynomial(boundary_data_matrix(f, d, o))
    n = length(f) - 1
    return [[P(j / n) for j in (-n):(-1)]; f]
end

function fc_coeff(f, d, o, k = length(f))
    fc = ext_func(f, d, o)
    n = length(fc)
    c = rfft(fc)[1:k]
    c .*= 2 / n
    c[1] /= 2
    return c
end

# Fourier series interpolation without continuation that shows gibbs ringing artifacts
function Gibbs_FC(f, k = length(f) >>> 1 + 1)
    n = length(f)
    c = rfft(f)[1:k]
    c .*= 2 / n
    c[1] /= 2
    p = Polynomial(c) # trigonometric interpolation polynomial
    return x -> real(p(exp(2π * im * (n - 1) * x / n)))
end

# Fourier series interpolation with continuation using Hermite polynomial with derivative of order d and approximation order o
function Hermite_FC(f, d, o, k = length(f), c = fc_coeff(f, d, o, k), p = Polynomial(c))
    n = 2 * (length(f) - 1) + 1 # length of the continuation
    return x -> real(p(exp(π * im * (n - 1) * (x + 1) / n)))
end