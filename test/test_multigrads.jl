
module XStates

   using Base: NamedTuple
using ACE, StaticArrays

   import Base: *, +, -, zero, rand, randn, show, promote_rule, rtoldefault, isapprox
   import LinearAlgebra: norm, promote_leaf_eltypes

   abstract type XState{SYMS, TT} <: ACE.AbstractState end 

   struct State{SYMS, TT} <: XState{SYMS, TT}
      x::NamedTuple{SYMS, TT}

      State{SYMS, TT}(t::NamedTuple{SYMS, TT1}) where {SYMS, TT, TT1} = new{SYMS, TT1}(t)
   end

   struct DState{SYMS, TT} <: XState{SYMS, TT}
      x::NamedTuple{SYMS, TT}

      DState{SYMS, TT}(t::NamedTuple{SYMS, TT1}) where {SYMS, TT, TT1} = new{SYMS, TT1}(t)
   end


   # some basic examples
   PosState{T} = State{(:rr,), Tuple{SVector{3, T}}}
   PosScalState{T} = State{(:rr, :x), Tuple{SVector{3, T}, T}}
   DPosScalState{T} = DState{(:rr, :x), Tuple{SVector{3, T}, T}}

   for f in (:zero, :rand, :randn) 
      eval( quote 
         function $f(::Union{TX, Type{TX}}) where {TX <: XState{SYMS, TT}} where {SYMS, TT} 
            vals = ntuple(i -> $f(TT.types[i]), length(SYMS))
            return TX( NamedTuple{SYMS}( vals ) )
         end
      end )
   end

   const _showdigits = 4
   _2str(x) = string(x)
   _2str(x::AbstractFloat) = "[$(round(x, digits=_showdigits))]"
   _2str(x::Complex) = "[$(round(x, digits=_showdigits))]"
   _2str(x::SVector{N, <: AbstractFloat}) where {N} = string(round.(x, digits=_showdigits))
   _2str(x::SVector{N, <: Complex}) where {N} = string(round.(x, digits=_showdigits))[11:end]

   _showsym(X::State) = ""
   _showsym(X::DState) = "'"

   show(io::IO, X::XState{SYMS}) where {SYMS} = 
         print(io, "{" * prod( "$(sym)$(_2str(getproperty(X.x, sym))), " 
                               for sym in SYMS) * "}" * _showsym(X))

   for f in (:+, :-)
      eval( quote 
         function $f(X1::TX1, X2::TX2) where {TX1 <: XState{SYMS}, TX2 <: XState{SYMS}} where {SYMS}
            vals = ntuple( i -> $f( getproperty(X1.x, SYMS[i]), 
                                    getproperty(X2.x, SYMS[i]) ), length(SYMS) )
            return TX1( NamedTuple{SYMS}(vals) )
         end
      end )
   end

   function *(X1::TX, a::Number) where {TX <: XState{SYMS}} where {SYMS}
      vals = ntuple( i -> *( getproperty(X1.x, SYMS[i]), a ), length(SYMS) )
      return TX( NamedTuple{SYMS}(vals) )
   end

   function *(a::Number, X1::TX) where {TX <: XState{SYMS}} where {SYMS}
      vals = ntuple( i -> *( getproperty(X1.x, SYMS[i]), a ), length(SYMS) )
      return TX( NamedTuple{SYMS}(vals) )
   end


   promote_leaf_eltypes(X::XState{SYMS}) where {SYMS} = 
      promote_type( ntuple(i -> promote_leaf_eltypes(getproperty(X.x, SYMS[i])), length(SYMS))... )

   norm(X::XState{SYMS}) where {SYMS} = 
         sum( norm( getproperty(X.x, sym) for sym in SYMS )^2 )

   isapprox(X1::TX, X2::TX, args...; kwargs...
            ) where {TX <: XState{SYMS}} where {SYMS} = 
      all( isapprox( getproperty(X1.x, sym), getproperty(X2.x, sym), 
                     args...; kwargs...) for sym in SYMS )
end 
##

zero(Main.XStates.PosState{Float64})
rand(Main.XStates.PosScalState{Float64})
randn(Main.XStates.PosScalState{ComplexF64})

@btime zero($(Main.XStates.PosScalState{Float64}))

X1 = rand(Main.XStates.PosScalState{Float64})
X2 = rand(Main.XStates.PosScalState{Float64})
X3 = rand(Main.XStates.PosScalState{ComplexF64})

Y1 = X1 + X2 
Y2 = X1 - X2
Y1 + Y2 ≈ 2 * X1
[Y1 + Y2] ≈ [2 * X1]

(1.2+2.3*im) * X1
X1 + X3

##

module NLMK 

using StaticArrays
using ACE, StaticArrays

import Base: *, +, -, zero, rand, show, promote_rule, rtoldefault
import LinearAlgebra: norm

struct NLMKState{T} <: ACE.AbstractState
   rr::SVector{3, T}
   x::T
end

NLMKState{T}(p::NamedTuple) where {T} = NLMKState(p) 
NLMKState(p::NamedTuple) = NLMKState(; p...)

NLMKState(T1 = Float64; rr::SVector{3, T} = zero(SVector{3, T1}), 
                        x::S = zero(T1))  where {T, S} =  
      NLMKState{promote_type(T, S)}(rr, x)

zero(::Type{NLMKState{T}}) where {T} = NLMKState(T)

rand(::Type{NLMKState}) = rand(NLMKState{Float64})
rand(::Type{NLMKState{T}}) where {T} = NLMKState(rand(SVector{3, T}), rand(T))

show(io::IO, s::NLMKState) = print(io, "{𝐫$(s.rr),x[$(s.x)]}")

promote_rule(::Type{T}, ::Type{NLMKState{S}}) where {T <: Number, S <: Number} = 
      NLMKState{promote_type(T, S)}

promote_rule(::Type{NLMKState{T}}, ::Type{NLMKState{S}}) where {T <: Number, S <: Number} = 
      NLMKState{promote_type(T, S)}

*(X::NLMKState, a::Number) = NLMKState(; rr = X.rr * a, x = X.x * a)

+(X1::NLMKState, X2::NLMKState) = NLMKState(; rr = X1.rr + X2.rr, x = X1.x + X2.x)
-(X1::NLMKState, X2::NLMKState) = NLMKState(; rr = X1.rr - X2.rr, x = X1.x - X2.x)

rtoldefault(::Union{T, Type{T}}, ::Union{T, Type{T}}, ::Real) where {T <: NLMKState{S}} where {S} =
      rtoldefault(real(S))

norm(X::NLMKState) = sqrt(norm(X.rr)^2 + abs(X.x)^2)

end 
##


@testset "Experimental Multi Grads" begin 

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      EuclideanVectorState, Product1pBasis, Scal1pBasis
using Random: shuffle
using ACEbase.Testing: fdtest, print_tf

##

maxdeg = 5
r0 = 1.0 
rcut = 3.0 
trans = trans = PolyTransform(1, r0)
Pk = ACE.scal1pbasis(:x, :k, maxdeg, trans, rcut)
RnYlm = ACE.Utils.RnYlm_1pbasis()

B1p = RnYlm * Pk
ACE.init1pspec!(B1p, maxdeg = maxdeg, Deg = ACE.NaiveTotalDegree())
length(B1p)

##


X = rand(Main.NLMK.NLMKState)
cfg = ACEConfig([ rand(Main.NLMK.NLMKState) for _=1:10 ])

Rn = B1p.bases[1]
Ylm = B1p.bases[2]
Pk = B1p.bases[3]

A = evaluate(B1p, cfg)
dA = evaluate_d(B1p, cfg)
A1, dA1 = ACE.evaluate_ed(B1p, cfg)


println(@test( A ≈ A1 ))
println(@test( dA ≈ dA1 ))

##

# gradient test 

_vec2NLMK(x) = Main.NLMK.NLMKState(rr = SVector{3}(x[1:3]), x = x[4])
_vec2NLMK_cfg(x) = ACEConfig( [_vec2NLMK(x)] )
_NLMK2vec(X) = [X.rr; [X.x]]

for ntest = 1:30
   x0 = randn(4)
   c = rand(length(B1p))
   F = x -> sum(ACE.evaluate(B1p, _vec2NLMK(x)) .* c)
   dF = x -> _NLMK2vec( sum(ACE.evaluate_d(B1p, _vec2NLMK_cfg(x)) .* c))
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()


##

end