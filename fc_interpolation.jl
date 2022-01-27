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

# save finite difference stencil up to order 4 with accuracy 4
const ForwardStencilMatrix = [forward_fdm(d, o) for d in 1:4, o in 1:4]
const BackwardStencilMatrix = [backward_fdm(d, o) for d in 1:4, o in 1:4]

function boundary_data_matrix(f, d, o)
    n = length(f) - 1
    F = zeros(2, d + 1)
    F[1, 1] = f[1]
    F[2, 1] = f[end]
    if (d > 4) || (o > 4)
        for j in 1:d
            F[1, j + 1] = ((n + 1) ^ j) * dot(forward_fdm(j, o), view(f, forward_stencil(j, o) .+ 1))
            F[2, j + 1] = ((n + 1) ^ j) * dot(backward_fdm(j, o), view(f, backward_stencil(j, o) .+ (n + 1)))
        end 
    else
        for j in 1:d
            F[1, j + 1] = ((n + 1) ^ j) * dot(ForwardStencilMatrix[j, o], view(f, forward_stencil(j, o) .+ 1))
            F[2, j + 1] = ((n + 1) ^ j) * dot(BackwardStencilMatrix[j, o], view(f, backward_stencil(j, o) .+ (n + 1)))
        end
    end
    return F
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

# save P0 and P1 polynomials
const P0Polynomials = [P0(m, r) for m in 0:4, r in 1:4]
const P1Polynomials = [P1(m, r) for m in 0:4, r in 1:4]

#=function boundary_data_matrix(f)
    n = length(f)
    return [f[1] (f[2] - f[1]) * n;
            f[end] (f[end] - f[end - 1]) * n]
end

function continuation_polynomial(F)
    return F[1, 1] * Polynomial([1.0; 0.0; -3.0; -2.0]) +
            F[2, 1] * Polynomial([0.0; 0.0; 3.0; 2.0]) +
            F[1, 2] * Polynomial([0.0; 1.0; 2.0; 1.0]) + 
            F[2, 2] * Polynomial([0.0; 0.0; 1.0; 1.0])
end=#

function continuation_polynomial(F)
    r = size(F, 2) - 1
    if r > 4
        p = F[1, 1] * P0(0, r) + F[2, 1] * P1(0, r)
        for m in 1:r
            p += F[1, m + 1] * P0(m, r) + F[2, m + 1] * P1(m, r)
        end
    else
        p = F[1, 1] * P0Polynomials[1, r] + F[2, 1] * P1Polynomials[1, r]
        for m in 1:r
            p += F[1, m + 1] * P0Polynomials[m + 1, r] + F[2, m + 1] * P1Polynomials[m + 1, r]
        end
    end
    return p
end

# extend the function from [0, 1] to [-1, 1] using 2 points Hermite interpolation with d the derivative order and o the approximation order
function ext_func(f, d, o)
    P = continuation_polynomial(boundary_data_matrix(f, d, o))
    n = length(f)
    return [f; [P(j / n) for j in (1 - n):(-1)]]
end

function fc_coeff(f, d, o)
    # f is defined on 0:(1 / n):((n - 1) / n)
    fc = ext_func(f, d, o) # continuation of f on 0:(1 / (2n - 1)):((2 * (n - 1)) / (2n - 1))
    c::Vector{Complex} = rfft(fc) / length(fc)
    c[1] /= 2
    return c
end

# Fourier continuation trigonometric interpolation polynomial
struct TrigoPolyFC
    p::Polynomial{Complex}
    shift::Float64
    order::Int
    accuracy::Int
    TrigoPolyFC(p::Polynomial, s::Float64, d::Int = 1, o::Int = 1) = new(p, s, d, o)
    function TrigoPolyFC(f::Vector{T}, d::Int = 1, o::Int = 1) where T <: Real
        if (d > 4) || (o > 4)
            @warn "High order can lead to numerical overflow"
        end
        n = (length(f) >>> - 1) - 1 # length of the continuation
        return new(Polynomial(fc_coeff(f, d, o)), (n + 1) / (2 * n), d, o) # x -> ((n + 1) / 2n) * x map [0.0; 1.0] to [0.0; 0.5 + 1 / n] with n = length(f)
    end
end
function (p::TrigoPolyFC)(x)
    return 2 * real(p.p(exp(2π * im * x * p.shift)))
end

# evaluate the interpolation Polynomial with up to k coefficients
function (p::TrigoPolyFC)(x, k::Int = length(p.p))
    ptrunc = Polynomial(view(p.p.coeffs, 1:k))
    return 2 * real(ptrunc(exp(2π * im * x * p.shift)))
end

Base.:+(p1::TrigoPolyFC, p2::TrigoPolyFC) = begin
    return TrigoPolyFC(p1.p + p2.p, (p1.shift + p2.shift) / 2, min(p1.order, p2.order), min(p1.accuracy, p2.accuracy))
end
Base.:-(p1::TrigoPolyFC, p2::TrigoPolyFC) = begin
    return TrigoPolyFC(p1.p - p2.p, (p1.shift + p2.shift) / 2, min(p1.order, p2.order), min(p1.accuracy, p2.accuracy))
end
Base.:*(i::Number, p::TrigoPolyFC) = begin
    return TrigoPolyFC(p.p * i, p.shift, p.order, p.accuracy)
end
Base.:*(p::TrigoPolyFC, i::Number) = Base.:*(i::Number, p::TrigoPolyFC)
Base.:/(p::TrigoPolyFC, i::Number) = begin
    return TrigoPolyFC(p.p / i, p.shift, p.order, p.accuracy)
end

function Polynomials.derivative(p::TrigoPolyFC, order::Int = 1)
    if p.order <= order
        @warn "Hermite interpolation order is $(p.order) lower or equal to $order"
    end

    c = similar(p.p.coeffs)
    for i in 1:length(c)
        c[i] = p.p.coeffs[i] * (2π * im * p.shift * (i - 1)) ^ order
    end
    return TrigoPolyFC(Polynomial(c), p.shift, p.order - order, p.accuracy)
end