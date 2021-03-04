module Mod
import Agents
import DataFrames
import StatsBase
import Distributions
# import InteractiveDynamics

mutable struct Forager <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int
    α::Float64 # learning rate
    k::Float64 # discount factor
    ρ::Float64 # utility exponent
    # utility U = V^ρ/(1+k*d) where d is distance
    β::Float64 # softmax temperature
    # nQ = Q./maximum(Q)
    # P(i) = exp(β⋅nQ[i])/∑exp(β⋅nQ[i])
    Q::Vector{Float64}
    creward::Float64 # cumulative reward
    cdistance::Float64 # cumulative travelled distance
end
function forager_init!(pos, gid, model; 
        μ_logα=0, σ_logα=2,
        μ_k=1,
        μ_logρ=log(0.9), σ_logρ=.15,
        μ_logβ=0.72, σ_logβ=0.8
        )
    αs = @. exp(μ_logα + $randn(model.forager_n) * σ_logα) # log(α) ∼ 𝒩 (μ_α, σ_α)
    ks = rand(Distributions.Exponential(μ_k), model.forager_n) # k ∼ exp (μ_k)
    ρs = rand(Distributions.LogNormal(μ_logρ,σ_logρ), model.forager_n) # ρ ∼log 𝒩 (μ_logρ, σ_logρ)
    βs = rand(Distributions.LogNormal(μ_logβ,σ_logβ), model.forager_n) # β ∼ 𝒰 (0.1, 10)
    for i in 1:model.forager_n
	    Agents.add_agent!(pos, Forager, model, gid, αs[i], ks[i], ρs[i], βs[i], zeros(Int, model.patch_n), 0.0, 0.0)
    end
end
function agent_step!(forager::Forager, model::Agents.ABM)
    # choose patch based on Q
    # Add normalization
    softmax_Q = @. exp(forager.β * forager.Q)
    softmax_Q ./= sum(softmax_Q)
    chosen_patch = StatsBase.sample(StatsBase.pweights(softmax_Q))
    model[chosen_patch].visit_counts[forager.gid] += 1
    forager.cdistance += Agents.edistance(forager, model[chosen_patch], model)
end

mutable struct Patch <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    σ_walk::Float64
    μ::Float64 # μ = ∑X_i where X_i ∼ 𝒩 (0, σ_walk)
    σ::Float64 # reward ∼ 𝒩 (μ, σ)
    visit_counts::Vector{Int}
end
function patch_init!(model; σ_walk=1, μ=10, σ=1)
    for _ in 1:model.patch_n
	# random positions
	Agents.add_agent!(Patch, model, σ_walk, σ*randn(1)+μ, σ, zeros(Int, model.forager_grp_n))
    end
end
function agent_step!(patch::Patch, model::Agents.ABM)
    reward = patch.μ + randn() * patch.σ
    
    softmax_visit_counts = exp.(3.*patch.visit_counts)
    softmax_visit_counts ./= sum(softmax_visit_counts)
    forager_grp = StatsBase.sample(StatsBase.pweights(softmax_visit_counts))
    # iterate over foragers of the chosen group
    for i in model.patch_n+(forager_grp-1)*model.forager_n+1:model.patch_n+(forager_grp)*model.forager_n
	forager = model[i]
	forager.creward += reward # only accumulate reward for foragers at this patch. 
	U = reward ^ forager.ρ / (1 + forager.k * Agents.edistance(forager, patch, model))
	forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
    end
    patch.μ += randn() * patch.σ_walk
    patch.visit_counts .= 0
end

# function model_step!(model::Agents.ABM)
# end

function model_init(; grid=(100, 100), forager_grp_n=2, forager_n=30, patch_n=30)
    # add patches first, but activate them last
    model = Agents.ABM(
		       Union{Forager, Patch},
		       Agents.GridSpace(grid);
		       scheduler=Agents.by_type((Forager, Patch), false),
		       properties=Dict{Symbol, Int}(:forager_grp_n=>forager_grp_n, :forager_n=>forager_n, :patch_n=>patch_n)
		       )
    patch_init!(model)
    forager_init!((1, 1), 1, model)
    forager_init!(grid, 2, model)
    model
end

function main(steps=1000)
    model = model_init()
    patch_μ = DataFrames.DataFrame(step=Int[], id=Int[], μ=Float64[])
    forager_Q = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], creward=Float64, cdistance=Float64, Q=Vector{Float64}[])
    for step in 1:steps
	Agents.step!(model, agent_step!, 1)
	for i in 1:model.patch_n
	    push!(patch_μ, (step, i, model[i].μ))
	end
	for i in 1:model.forager_n
	    push!(forager_Q, (step, i, 1, model[model.patch_n+i].creward, model[model.patch_n+i].cdistance, copy(model[model.patch_n+i].Q)))
	    push!(forager_Q, (step, i, 2, model[model.patch_n+model.forager_n+i].creward, model[model.patch_n+model.forager_n+i].cdistance, copy(model[model.patch_n+model.forager_n+i].Q)))
	end
    end
    patch_μ, forager_Q
    # InteractiveDynamics.abm_plot(model; ac)
end
end
