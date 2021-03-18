# helper functions
function mn_softmax_sample(x, β) # maximum-normalized
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
function edis(a, b)
    (a .- b) .^ 2 |> sum |> sqrt
end
function argmaxes(itr)
    m = maximum(itr)
    filter(i -> itr[i] == m, 1:length(itr))
end

mutable struct Forager0 <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int # group id
    α::Float64 # learning rate; 0 < α < 1; α↑ => discount previous learning
    k::Float64 # discount factor; k > 1; k↑ => distance averse
    ρ::Float64 # utility exponent; ρ > 0; ρ > 1 => risk seeking
    β::Float64 # softmax temperature; β↑ => exploit↑, explore↓
    chosen_patch::Int # chosen patch at current step = mn_softmax_sample(Q, β)
    rew::Float64 # reward at current step; rew == 0.0 => failure
    dis::Float64 # travelled distance at current step
    Q::Vector{Float64} # learned utility of every patch
    U::Vector{Float64} # real utility of every patch
    # U = V^ρ/(1+k*distance)
end
mutable struct Patch0 <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    σ_walk::Float64
    μ_rew::Float64 # μ_rew = ∑X_i where X_i ∼ 𝒩 (0, σ_walk)
    σ_rew::Float64 # reward ∼ 𝒩 (μ_rew, σ_rew)
    visited_by::Vector{Vector{Forager0}} # a vector of foragers for every forager group
end
function model_step!(model::Agents.ABM)
    # activate foragers
    for forager in vcat(model.foragers...)
        # choose patch based on softmax of maximum-normalized Q
        chosen_patch = mn_softmax_sample(forager.Q, forager.β)
        forager.chosen_patch = chosen_patch
        patch = model[chosen_patch]
        # visit chosen patch
        push!(patch.visited_by[forager.gid], forager)
        forager.dis = edis(forager.pos, patch.pos)
    end
    # activate patches
    for patch in model.patches
        visit_counts = map(length, patch.visited_by)
        if sum(visit_counts) != 0
            # choose forager group to award food based on the number of visitors of that group
            chosen_forager_grp = StatsBase.sample(argmaxes(visit_counts))
            rew = rand(Distributions.Normal(patch.μ_rew, patch.σ_rew))
            # set forager.rew for visiting foragers
            # update Q[patch.id] for foragers whose group members visited patch
            for (gid, gforagers) in enumerate(model.foragers)
                if gid == chosen_forager_grp
                    d = patch.visited_by[gid][1].dis
                    for forager in patch.visited_by[gid]
                        forager.rew = rew/visit_counts[gid]
                        U = symmetrical_utility(forager.rew, forager.ρ, forager.k, d)
                        forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    end
                    # distance is the same for foragers of the same group
                    # for forager in gforagers
                    #     U = symmetrical_utility(forager.rew, forager.ρ, forager.k, d)
                    #     forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    # end
                    # cleanup
                    empty!(patch.visited_by[gid])
                elseif visit_counts[gid] > 0
                    for forager in patch.visited_by[gid]
                        forager.rew = 0.0
                        U = 0.0
                        forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    end
                    # for forager in gforagers
                    #     U = 0.0
                    #     forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    # end
                    # cleanup
                    empty!(patch.visited_by[gid])
                end
                # update U[patch.id] of all foragers
                d = edis(gforagers[1].pos, patch.pos)
                for forager in gforagers
                    forager.U[patch.id] = symmetrical_utility(patch.μ_rew, forager.ρ, forager.k, d)
                end
            end
        end
        # random walk of μ_rew
        patch.μ_rew += rand(Distributions.Normal(0.0, patch.σ_walk))
    end
end
function vicarious_model_step!(model::Agents.ABM)
    # activate foragers
    for forager in vcat(model.foragers...)
        # choose patch based on softmax of maximum-normalized Q
        chosen_patch = mn_softmax_sample(forager.Q, forager.β)
        forager.chosen_patch = chosen_patch
        patch = model[chosen_patch]
        # visit chosen patch
        push!(patch.visited_by[forager.gid], forager)
        forager.dis = edis(forager.pos, patch.pos)
    end
    # activate patches
    for patch in model.patches
        visit_counts = map(length, patch.visited_by)
        if sum(visit_counts) != 0
            # choose forager group to award food based on the number of visitors of that group
            chosen_forager_grp = StatsBase.sample(argmaxes(visit_counts))
            rew = rand(Distributions.Normal(patch.μ_rew, patch.σ_rew))
            # set forager.rew for visiting foragers
            # update Q[patch.id] for foragers whose group members visited patch
            for (gid, gforagers) in enumerate(model.foragers)
                if gid == chosen_forager_grp
                    d = patch.visited_by[gid][1].dis
                    for forager in patch.visited_by[gid]
                        forager.rew = rew/visit_counts[gid]
                    end
                    # vicarious learning
                    for forager in gforagers
                        U = symmetrical_utility(forager.rew, forager.ρ, forager.k, d)
                        forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    end
                    # cleanup
                    empty!(patch.visited_by[gid])
                elseif visit_counts[gid] > 0
                    for forager in patch.visited_by[gid]
                        forager.rew = 0.0
                    end
                    # vicarious learning
                    for forager in gforagers
                        U = 0.0
                        forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                    end
                    # cleanup
                    empty!(patch.visited_by[gid])
                end
                # update U[patch.id] of all foragers
                d = edis(gforagers[1].pos, patch.pos)
                for forager in gforagers
                    forager.U[patch.id] = symmetrical_utility(patch.μ_rew, forager.ρ, forager.k, d)
                end
            end
        end
        # random walk of μ_rew
        patch.μ_rew += rand(Distributions.Normal(0.0, patch.σ_walk))
    end
end

function init_model0(; grid=(100, 100),
        patch_n=30,
        σ_walk=1.0,
        μ_μ_rew=10.0,
        σ_μ_rew=1.0,
        σ_rew=1.0,
        forager_ns=[30, 30],
        poss=[(1, 1), grid],
        μ_logαs=[log(0.1), log(0.1)], σ_logαs=[0.3, 0.3],
        μ_ks=[1.0, 1.0],
        μ_logρs=[log(0.9), log(0.9)], σ_logρs=[0.15, 0.15],
        μ_logβs=[log(0.72), log(0.72)], σ_logβs=[0.8, 0.8]
    )
    model = Agents.ABM(
               Union{Forager0, Patch0},
               Agents.GridSpace(grid);
               properties=Dict(
                               :patches=>Vector{Patch0}(undef, patch_n),
                               :foragers=>[Vector{Forager0}(undef, forager_n) for forager_n in forager_ns]
                              ),
               warn=false
               )
    # add patches
    μ_rews = rand(Distributions.Normal(μ_μ_rew, σ_μ_rew), patch_n)
    for i in 1:patch_n
        # random positions
        model.patches[i] = Agents.add_agent!(Patch0, model, σ_walk, μ_rews[i], σ_rew, [Forager0[] for _ in 1:length(forager_ns)])
    end
    # add foragers
    for (gid, forager_n) in enumerate(forager_ns)
        αs = rand(Distributions.LogNormal(μ_logαs[gid], σ_logαs[gid]), forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
        ks = rand(Distributions.Exponential(μ_ks[gid]), forager_n) # k ∼ Exp(μ_k)
        ρs = rand(Distributions.LogNormal(μ_logρs[gid], σ_logρs[gid]), forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
        βs = rand(Distributions.LogNormal(μ_logβs[gid], σ_logβs[gid]), forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
        # TODO k = 0.0 for now
        for i in 1:forager_n
            model.foragers[gid][i] = Agents.add_agent!(poss[gid], Forager0, model, gid, αs[i], 0.0, ρs[i], βs[i], 0, 0.0, 0.0, zeros(patch_n).+eps(), zeros(patch_n))
        end
    end
    model
end

function collect_model0(model; vicarious=true, steps=1000)
    patch_static = DataFrames.DataFrame(id=Int[], σ_walk=Float64[], σ_rew=Float64[])
    patch_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], μ_rew=Float64[])
    forager_static = DataFrames.DataFrame(id=Int[], α=Float64[], k=Float64[], ρ=Float64[], β=Float64[])
    forager_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], chosen_patch=Int[], rew=Float64[], dis=Float64[], Q=Vector{Float64}[], U=Vector{Float64}[])
    for patch in model.patches
        push!(patch_static, (patch.id, patch.σ_walk, patch.σ_rew))
    end
    for forager in vcat(model.foragers...)
        push!(forager_static, (forager.id, forager.α, forager.k, forager.ρ, forager.β))
    end
    for step in 1:steps
        if vicarious
            Agents.step!(model, Agents.dummystep, vicarious_model_step!, 1)
        else
            Agents.step!(model, Agents.dummystep, model_step!, 1)
        end
        for patch in model.patches
            push!(patch_dynamic, (step, patch.id, patch.μ_rew))
        end
        for forager in vcat(model.foragers...)
            push!(forager_dynamic, (step, forager.id, forager.gid, forager.chosen_patch, forager.rew, forager.dis, copy(forager.Q), copy(forager.U)))
        end
    end
    patch_static, forager_static, patch_dynamic, forager_dynamic
end
