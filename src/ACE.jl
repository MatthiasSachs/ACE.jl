
module ACE

using Base: NamedTuple
using Reexport

include("objectpools.jl")
using ACE.ObjectPools: acquire!, release!
const _pool = ObjectPools.ArrayPool()

# external imports that are useful for all submodules
include("imports.jl")

@extimports
@baseimports

# TODO - move to imports

using ForwardDiff: derivative
import ChainRules: rrule, NO_FIELDS, NoTangent
import ACE: evaluate, evaluate_d 


abstract type AbstractACEModel end 

abstract type AbstractProperty end

function coco_init end
function coco_zeros end
function coco_filter end
function coco_dot end

# TODO 
# * decide on rand(basis) interface
# * move these the following definitions to ACEbase
function _rrule_evaluate end 
function _rrule_evaluate_d end 
import  ACEbase: gradtype, valtype, alloc_B, alloc_dB

include("auxiliary.jl")


include("rotations3d.jl")
using ACE.Rotations3D
include("testing/wigner.jl")


include("states.jl")
include("properties.jl")


include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using ACE.Transforms
include("polynomials/orthpolys.jl"); @reexport using ACE.OrthPolys

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

# three specific 1p-bases that are useful
include("Ylm1pbasis.jl")
include("Rn1pbasis.jl")
include("scal1pbasis.jl")

include("product_1pbasis.jl")

include("sparsegrids.jl")


# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
include("pibasis.jl")

include("symmbasis.jl")

# models and model evaluators 


include("linearmodel.jl")

include("evaluator.jl")
# include("grapheval.jl")

# include("linearmodel.jl")


# # orthogonal basis
# include("orth.jl")
#
# lots of stuff related to random samples:
#  - random configurations
#  - random potentials
#  ...
include("random.jl")
@reexport using ACE.Random


include("utils.jl")
@reexport using ACE.Utils

# include("utils/importv5.jl")
# include("compat/compat.jl")
# include("export/export.jl")


include("testing/testing.jl")

# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials
# TODO -> move elsewhere!!!
# include("rpi/rpi_degrees.jl")

include("ad.jl")

include("models/models.jl")

end # module
