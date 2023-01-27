module XiruiModels

import Agents
import Agents.Pathfinding
using DataFrames
using DataFramesMeta
import Distributions
import Plots
import Formatting
import Random
import Statistics
import StatsBase
using StatsPlots
# import InteractiveDynamics

include("helper_functions.jl")
#=
function argmaxes(x::Vector)::Vector{Int}
function edis(a, b)::Float64 # euclidean distance
function entropy(x::Vector{T})::Float64 where T
function proportion(patch_ids, patch_n)::Vector{Float64}
function mn_softmax_sample(x::Vector{T}, β)::Int where T # maximum-normalized
function sym_utility_risk(V, ρ)::Float64
=#
include("modelA.jl")
#
#
#
#=
# gridspace is not used
function init_modelA(
    forager_ns=[30, 30],
    μ_logαs=[log(0.1), log(0.1)], σ_logαs=[0.3, 0.3],
    μ_logρs=[log(0.9), log(0.9)], σ_logρs=[0.15, 0.15],
    μ_logβs=[log(0.72), log(0.72)], σ_logβs=[0.8, 0.8],
    patch_n=30,
    μ_base=10, σ_base=3,
    capacity=sum(forager_ns)/2,
    decay=1.0,
)::Agents.ABM
function collect_modelA(model; steps=1000)::NTuple{3, DataFrames.DataFrame} # patch_static, forager_static, forager_dynamic, model
function shock_modelA(model; steps=1000, μ_base=10, σ_base=10)::NTuple{3, DataFrames.DataFrame}
=#
# include("modelB.jl")
#=
# gridspace space is used
# move via Agents.Pathfinding
# agent state transitions home, return, leave, forage
# patches grow by τ, what's the reward? 10%?
# A, grows back τ, eated percentage β
function init_modelB()::Agents.ABM
function collect_modelB()::NTuple{4, DataFrames.DataFrame}
=#
end
