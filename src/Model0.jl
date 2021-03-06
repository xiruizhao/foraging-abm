# util functions
function softmax_sample(x, β)
    y = exp.(β .* x)
    y ./= sum(y)
    StatsBase.sample(StatsBase.pweights(y))
end
function mn_softmax_sample(x, β)
    # softmax of maximum-normalized softmax
    y = x ./ maximum(x)
    y .= exp.(β .* x)
    y ./= sum(y)
    StatsBase.sample(StatsBase.pweights(y))
end
function symmetrical_utility(V::Float64, ρ, k, d)
    if V < 0.0
        -symmetrical_utility(-V, ρ, k, d)
    else
        V^ρ/(1.0+k*d)
    end
end

mutable struct Forager0 <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int # group id
    α::Float64 # learning rate
    k::Float64 # discount factor
    ρ::Float64 # utility exponent
    # utility U = V^ρ/(1+k*d) where d is distance
    β::Float64 # softmax temperature
    chosen_patch::Int
    rew::Float64 # reward at current step
    dis::Float64 # travelled distance at current step
    Q::Vector{Float64}
    target_Q::Vector{Float64}
    # nQ = Q./maximum(Q)
    # P(i) = exp(β⋅nQ[i])/∑exp(β⋅nQ[i])
end
function add_forager0!(model; pos, gid, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
    αs = rand(Distributions.LogNormal(μ_logα, σ_logα), model.forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
    ks = rand(Distributions.Exponential(μ_k), model.forager_n) # k ∼ Exp(μ_k)
    ρs = rand(Distributions.LogNormal(μ_logρ,σ_logρ), model.forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
    βs = rand(Distributions.LogNormal(μ_logβ,σ_logβ), model.forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
    for i in 1:model.forager_n
        target_Q = zeros(model.patch_n)
        for j in 1:model.patch_n
            patch = model[j]
            target_Q[j] = symmetrical_utility(patch.μ_rew, ρs[i], ks[i], Agents.edistance(patch.pos, pos, model))
        end
        Agents.add_agent!(pos, Forager0, model, gid, αs[i], ks[i], ρs[i], βs[i], 0, 0.0, 0.0, zeros(model.patch_n), target_Q)
    end
end
function agent_step!(forager::Forager0, model::Agents.ABM)
    # choose patch based on softmax of maximum-normalized Q
    forager.chosen_patch = mn_softmax_sample(forager.Q, forager.β)
    patch = model[forager.chosen_patch]
    push!(patch.visited_by[forager.gid], forager.id)
    forager.dis = Agents.edistance(forager, patch, model)
end

mutable struct Patch0 <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    σ_walk::Float64
    μ_rew::Float64 # μ_rew = ∑X_i where X_i ∼ 𝒩 (0, σ_walk)
    σ_rew::Float64 # reward ∼ 𝒩 (μ_rew, σ_rew)
    visited_by::Vector{Vector{Int}}
end
function add_patch0!(model; σ_walk, μ_μ_rew, σ_μ_rew, σ_rew)
    μ_rews = rand(Distributions.Normal(μ_μ_rew, σ_μ_rew), model.patch_n)
    for i in 1:model.patch_n
        # random positions
        Agents.add_agent!(Patch0, model, σ_walk, μ_rews[i], σ_rew, [Int[] for _ in 1:model.forager_grp_n])
    end
end
function agent_step!(patch::Patch0, model::Agents.ABM)
    # choose forager group to award food based on softmax of number of visitors of that group
    visit_counts = map(length, patch.visited_by)
    chosen_forager_grp = softmax_sample(visit_counts, 1.0)
    reward = rand(Distributions.Normal(patch.μ_rew, patch.σ_rew))
    # iterate over foragers of the chosen group
    for i in patch.visited_by[chosen_forager_grp]
        forager = model[i]
        forager.rew = reward/visit_counts[chosen_forager_grp]
        U = symmetrical_utility(forager.rew, forager.ρ, forager.k, Agents.edistance(forager, patch, model))
        forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
    end
    patch.μ_rew += rand(Distributions.Normal(0.0, patch.σ_walk))
    for i in 1:model.forager_grp_n
        empty!(patch.visited_by[i])
    end
    for i in model.patch_n+1:Agents.nagents(model)
        forager = model[i]
        forager.target_Q[patch.id] = symmetrical_utility(patch.μ_rew, forager.ρ, forager.k, Agents.edistance(patch, forager, model))
    end
end

# function model_step!(model::Agents.ABM)
# end

function init_model0(; grid=(100, 100),
        patch_n=30,
        σ_walk=1.0,
        μ_μ_rew=10.0,
        σ_μ_rew=1.0,
        σ_rew=1.0,
        forager_grp_n=2, forager_n=30,
        μ_logα=log(0.1), σ_logα=0.3,
        μ_k=1.0,
        μ_logρ=log(0.9), σ_logρ=0.15,
        μ_logβ=log(0.72), σ_logβ=0.8
    )
    # add patches first, but activate them last
    model = Agents.ABM(
               Union{Forager0, Patch0},
               Agents.GridSpace(grid);
               scheduler=Agents.by_type((Forager0, Patch0), false),
               properties=Dict(:forager_grp_n=>forager_grp_n, :forager_n=>forager_n, :patch_n=>patch_n),
               warn=false
               )
    add_patch0!(model; σ_walk, μ_μ_rew, σ_μ_rew, σ_rew)
    add_forager0!(model; pos=(1, 1), gid=1, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
    add_forager0!(model; pos=grid, gid=2, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
    model
end

# function init_model0(; grid=(100, 100),
#         patch_n=30,
#         σ_walk=1.0,
#         μ_μ_rew=10.0,
#         σ_μ_rew=1.0,
#         σ_rew=1.0,
#         forager_n=[30, 30],
#         μ_logα=[log(0.1), log(0.1)], σ_logα=[0.3, 0.3],
#         μ_k=[1.0, 1.0],
#         μ_logρ=log(0.9), σ_logρ=0.15,
#         μ_logβ=log(0.72), σ_logβ=0.8
#     )
#     # add patches first, but activate them last
#     model = Agents.ABM(
#                Union{Forager0, Patch0},
#                Agents.GridSpace(grid);
#                scheduler=Agents.by_type((Forager0, Patch0), false),
#                properties=Dict(:forager_grp_n=>forager_grp_n, :forager_n=>forager_n, :patch_n=>patch_n),
#                warn=false
#                )
#     add_patch0!(model; σ_walk, μ_μ_rew, σ_μ_rew, σ_rew)
#     add_forager0!(model; pos=(1, 1), gid=1, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
#     add_forager0!(model; pos=grid, gid=2, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
#     model
# end

function collect_model0(model; steps=1000)
    patch_static = DataFrames.DataFrame(id=Int[], σ_walk=Float64[], σ_rew=Float64[])
    patch_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], μ_rew=Float64[])
    forager_static = DataFrames.DataFrame(id=Int[], α=Float64[], k=Float64[], ρ=Float64[], β=Float64[])
    forager_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], chosen_patch=Int[], rew=Float64[], dis=Float64[], Q=Vector{Float64}[], target_Q=Vector{Float64}[])
    for patch_i in 1:model.patch_n
        patch = model[patch_i]
        push!(patch_static, (patch.id, patch.σ_walk, patch.σ_rew))
    end
    for forager_i in model.patch_n+1:Agents.nagents(model)
        forager = model[forager_i]
        push!(forager_static, (forager.id, forager.α, forager.k, forager.ρ, forager.β))
    end
    for step in 1:steps
        Agents.step!(model, agent_step!, 1)
        for patch_i in 1:model.patch_n
            patch = model[patch_i]
            push!(patch_dynamic, (step, patch.id, patch.μ_rew))
        end
        for forager_i in model.patch_n+1:Agents.nagents(model)
            forager = model[forager_i]
            push!(forager_dynamic, (step, forager.id, forager.gid, forager.chosen_patch, forager.rew, forager.dis, copy(forager.Q), copy(forager.target_Q)))
        end
    end
    patch_static, forager_static, patch_dynamic, forager_dynamic
end


# mutable struct Forager1 <: Agents.AbstractAgent
#     id::Int
#     pos::Dims{2}
#     gid::Int # group id
#     gstore::Ref{Float64} # pointer to group food store
#     α::Float64 # learning rate
#     k::Float64 # discount factor
#     ρ::Float64 # utility exponent
#     # utility U = V^ρ/(1+k*d) where d is distance
#     β::Float64 # softmax temperature
#     chosen_patch::Int
#     rew::Float64 # reward at current step
#     dis::Float64 # travelled distance at current step
#     Q::Vector{Float64}
#     target_Q::Vector{Float64}
#     # nQ = Q./maximum(Q)
#     # P(i) = exp(β⋅nQ[i])/∑exp(β⋅nQ[i])
# end
# function add_forager0!(model; pos, gid, μ_logα, σ_logα, μ_k, μ_logρ, σ_logρ, μ_logβ, σ_logβ)
#     αs = rand(Distributions.LogNormal(μ_logα, σ_logα), model.forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
#     ks = rand(Distributions.Exponential(μ_k), model.forager_n) # k ∼ Exp(μ_k)
#     ρs = rand(Distributions.LogNormal(μ_logρ,σ_logρ), model.forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
#     βs = rand(Distributions.LogNormal(μ_logβ,σ_logβ), model.forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
#     for i in 1:model.forager_n
#         target_Q = zeros(model.patch_n)
#         for j in 1:model.patch_n
#             patch = model[j]
#             target_Q[j] = symmetrical_utility(patch.μ_rew, ρs[i], ks[i], Agents.edistance(patch.pos, pos, model))
#         end
#         Agents.add_agent!(pos, Forager0, model, gid, αs[i], ks[i], ρs[i], βs[i], 0, 0.0, 0.0, zeros(model.patch_n), target_Q)
#     end
# end
# function agent_step!(forager::Forager0, model::Agents.ABM)
#     # choose patch based on softmax of maximum-normalized Q
#     forager.chosen_patch = mn_softmax_sample(forager.Q, forager.β)
#     patch = model[forager.chosen_patch]
#     push!(patch.visited_by[forager.gid], forager.id)
#     forager.dis = Agents.edistance(forager, patch, model)
# end
# function model1()
# end
