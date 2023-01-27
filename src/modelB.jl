@enum State home intransit1 intransit2
mutable struct ForagerB <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    home::Dims{2}
    gid::Int # group id
    α::Float64 # learning rate ∈(0, 1); α↑ => discount past experience
    # speed::Float64 # speed
    ρ::Float64 # utility exponent ∈(0, 1) => ρ↑ => risk tolerance↑; (1, ∞) irrational, excluded
    β::Float64 # softmax temperature; β↑ => exploit↑, explore↓
    mrate::Float64 # metabolic rate
    threshold::Float64 # reproduction threshold
    chosen_patch::Int # chosen patch at current step = mn_softmax_sample(Q, β)
    energy::Float64
    Q::Vector{Float64} # learned utility of every patch
    state::State
    # U = V^ρ - d * mrate
end
mutable struct PatchB <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    base::Float64
    val::Float64
    τ::Float64
    # at each step, if patch is being foraged, val *= 0.9
    # otherwise, val += (base-val)/τ
    # visited_by::Vector{Forager1}
end
function step_modelB!(model::Agents.ABM)
    # activate foragers
    for forager in vcat(model.foragers...) # filter(x->typeof(x)==Forager, )Agents.
        if forager.state == home
            # choose patch based on softmax of maximum-normalized Q
            chosen_patch = mn_softmax_sample(forager.Q, forager.β)
            forager.chosen_patch = chosen_patch
            Pathfinding.set_target!(forager, model.patches[chosen_patch].pos, model)
            forager.state = intransit1
            Agents.move_along_route!(forager, model)
        else if forager.state == intransit1
            patch = model.patches[forager.chosen_patch]
            if forager.pos == patch.pos
                rew = patch.val * 0.1
                U = sym_utility_risk(rew, forager.ρ)
                forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                # TODO vicarious
                forager.energy += rew
                patch.val *= 0.9
                Pathfinding.set_target!(forager, forager.home, model)
                forager.state = intransit2
            else
                Agents.move_along_route!(forager, model)
            end
        else
            if forager.pos == forager.home
                forager.state = home
            else
                Agents.move_along_route!(forager, model)
            end
        end
        forager.energy -= forager.mrate
        if forager.energy < 0
            Agents.kill_agent!(forager, model)
        else if forager.energy > forager.threshold
            # Agents.add_agent(Forager, forager.energy/2)
            forager.energy /= 2
        end
    end
    # activate patches
    for patch in model.patches
        patch.val += (patch.base - patch.val)/patch.τ
        #=
        visits_count = map(length, patch.visited_by)
        if sum(visits_count) > 0 # skip unvisited patches
            sorted_visited_by = sort!(patch.visited_by; by=x->x.speed_rank)
            faster_foragers_count = 0
            learning_store = [Float64[] for _ in 1:length(model.foragers)]
            for forager in sorted_visited_by
                forager.rew = patch.base/(1+exp(patch.decay*(faster_foragers_count - patch.capacity)))
                push!(learning_store[forager.gid], forager.rew)
                U = sym_utility_risk(forager.rew, forager.ρ)
                forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                faster_foragers_count += 1
            end
            # vicarious learning
            for (gid, rews) in enumerate(learning_store)
                if length(rews) > 0
                    mean_rew = Statistics.mean(rews)
                    for forager in model.foragers[gid]
                        if forager.chosen_patch != patch.id
                            U = sym_utility_risk(mean_rew, forager.ρ)
                            forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
                        end
                    end
                end
            end
            empty!(patch.visited_by)
        end
        =#
    end
end
function init_model1(;
    forager_ns=[30, 30],
    μ_logαs=[log(0.1), log(0.1)], σ_logαs=[0.3, 0.3],
    μ_logρs=[log(0.9), log(0.9)], σ_logρs=[0.15, 0.15],
    μ_logβs=[log(0.72), log(0.72)], σ_logβs=[0.8, 0.8],
    patch_n=30,
    μ_base=10, σ_base=3,
    τ=3
)
    model = Agents.ABM(
               Union{Forager1, Patch1},
               Agents.GridSpace((0,0));
               properties=Dict(
                               :patches=>Vector{Patch1}(undef, patch_n),
                               :foragers=>[Vector{Forager1}(undef, forager_n) for forager_n in forager_ns]
                              ),
               warn=false
               )
    # add patches
    bases = rand(Distributions.Normal(μ_base, σ_base), patch_n)
    for i in 1:patch_n
        model.patches[i] = Agents.add_agent!(Patch1, model, bases[i], capacity, decay, [Forager1[] for _ in 1:length(forager_ns)])
    end
    # add foragers
    speed_ranks = Random.randperm(sum(forager_ns))
    sri = 1
    for (gid, forager_n) in enumerate(forager_ns)
        αs = rand(Distributions.LogNormal(μ_logαs[gid], σ_logαs[gid]), forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
        ρs = rand(Distributions.LogNormal(μ_logρs[gid], σ_logρs[gid]), forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
        βs = rand(Distributions.LogNormal(μ_logβs[gid], σ_logβs[gid]), forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
        for i in 1:forager_n
            model.foragers[gid][i] = Agents.add_agent!(Forager1, model, gid, αs[i], speed_ranks[i], 0.0, ρs[i], βs[i], 0, 0.0, 0.0, zeros(patch_n).+eps(), zeros(patch_n))
            sri += 1
        end
    end
    model
end
function collect_model1(model; steps=1000)
    patch_static = DataFrames.DataFrame(id=Int[], base=Float64[])
    forager_static = DataFrames.DataFrame(id=Int[], α=Float64[], speed_rank=Int[], ρ=Float64[], β=Float64[], U=Vector{Float64}[]) # U is the real utility of every patch
    forager_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], chosen_patch=Int[], rew=Float64[], Q=Vector{Float64}[])
    for patch in model.patches
        push!(patch_static, (patch.id, patch.base))
    end
    for forager in vcat(model.foragers...)
        push!(forager_static, (forager.id, forager.α, forager.speed_rank, forager.ρ, forager.β, patch_static.base.^forager.ρ))
    end
    for step in 1:steps
        Agents.step!(model, Agents.dummystep, step_model1!, 1)
        for forager in vcat(model.foragers...)
            push!(forager_dynamic, (step, forager.id, forager.gid, forager.chosen_patch, forager.rew, copy(forager.Q)))
        end
    end
    patch_static, forager_static, forager_dynamic
end
function shock_model1(model; steps=1000)
    for patch in model.patches
        patch.base += 3 * rand()
    end
    collect_model1(model; steps)
end
