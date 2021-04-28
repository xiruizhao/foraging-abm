mutable struct ForagerA <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int # group id
    α::Float64 # learning rate ∈(0, 1); α↑ => discount past experience
    speed_rank::Int # speed ranking of foragers
    ρ::Float64 # utility exponent ∈(0, 1) => ρ↑ => risk tolerance↑; (1, ∞) irrational, excluded
    β::Float64 # softmax temperature; β↑ => exploit↑, explore↓
    chosen_patch::Int # chosen patch at current step = mn_softmax_sample(Q, β)
    rew::Float64 # reward at current step
    Q::Vector{Float64} # learned utility of every patch
    # U = V^ρ
end
mutable struct PatchA <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    base::Float64
    capacity::Float64 # midpoint
    decay::Float64 # reward decay rate
    # rew = base/(1 + exp(decay*(faster_foragers_count - capacity)))
    visited_by::Vector{ForagerA}
end
function step_modelA!(model::Agents.ABM)
    # activate foragers
    for forager in vcat(model.foragers...)
        # choose patch based on softmax of maximum-normalized Q
        chosen_patch = mn_softmax_sample(forager.Q, forager.β)
        forager.chosen_patch = chosen_patch
        patch = model.patches[chosen_patch]
        # visit chosen patch
        push!(patch.visited_by, forager)
    end
    # activate patches
    for patch in model.patches
        if length(patch.visited_by) > 0 # skip unvisited patches
            sort!(patch.visited_by; by=x->x.speed_rank)
            faster_foragers_count = 0
            learning_store = [Float64[] for _ in 1:length(model.foragers)]
            for forager in patch.visited_by
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
    end
end
function init_modelA(;
    forager_ns=[30, 30],
    μ_logαs=[log(0.1), log(0.1)], σ_logαs=[0.3, 0.3],
    μ_logρs=[log(0.9), log(0.9)], σ_logρs=[0.03, 0.03],
    μ_logβs=[log(0.72), log(0.72)], σ_logβs=[0.8, 0.8],
    patch_n=30,
    μ_base=10, σ_base=3,
    capacity=sum(forager_ns)/2,
    decay=1.0,
)
    model = Agents.ABM(
               Union{ForagerA, PatchA},
               Agents.GridSpace((1,1));
               properties=Dict(
                               :patches=>Vector{PatchA}(undef, patch_n),
                               :foragers=>[Vector{ForagerA}(undef, forager_n) for forager_n in forager_ns]
                              ),
               warn=false
               )
    # add patches
    bases = sort(rand(Distributions.Normal(μ_base, σ_base), patch_n))
    for i in 1:patch_n
        model.patches[i] = Agents.add_agent!((1,1), PatchA, model, bases[i], capacity, decay, ForagerA[])
    end
    # add foragers
    speed_ranks = Random.randperm(sum(forager_ns))
    sri = 1
    for (gid, forager_n) in enumerate(forager_ns)
        αs = rand(Distributions.LogNormal(μ_logαs[gid], σ_logαs[gid]), forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
        ρs = rand(Distributions.LogNormal(μ_logρs[gid], σ_logρs[gid]), forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
        βs = rand(Distributions.LogNormal(μ_logβs[gid], σ_logβs[gid]), forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
        for i in 1:forager_n
            model.foragers[gid][i] = Agents.add_agent!((1,1), ForagerA, model, gid, αs[i], speed_ranks[i], ρs[i], βs[i], 0, 0.0, ones(patch_n))
            sri += 1
        end
    end
    model
end
function collect_modelA(model; steps=1000)
    patch_static = DataFrames.DataFrame(id=Int[], base=Float64[])
    forager_static = DataFrames.DataFrame(id=Int[], α=Float64[], speed_rank=Int[], ρ=Float64[], β=Float64[], U=Vector{Float64}[]) # U is the real utility of every patch
    forager_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], chosen_patch=Int[], rew=Float64[], Q=Vector{Float64}[])
    for patch in model.patches
        push!(patch_static, (patch.id, patch.base))
    end
    U_scale = 1 + exp(model.patches[1].decay*(-model.patches[1].capacity))
    for forager in vcat(model.foragers...)
        push!(forager_static, (forager.id, forager.α, forager.speed_rank, forager.ρ, forager.β, (patch_static.base./U_scale).^forager.ρ))
    end
    for step in 1:steps
        Agents.step!(model, Agents.dummystep, step_modelA!, 1)
        for forager in vcat(model.foragers...)
            push!(forager_dynamic, (step, forager.id, forager.gid, forager.chosen_patch, forager.rew, copy(forager.Q)))
        end
    end
    patch_static, forager_static, forager_dynamic, model
end
function shock_modelA(model; steps=1000, μ_base=10, σ_base=3)
    for patch in model.patches
        patch.base = rand(Distributions.Normal(μ_base, σ_base))
    end
    collect_modelA(model; steps)
end
function modelA_heatmaps(ps, fs, fd)
    ps.id = map(string, ps.id)
    choice_plts = []
    Q_plts = []
    forager_grp_n = fd.gid[end]
    forager_n = nrow(fs)
    patch_n = nrow(ps)
    bin_n = 250 # 250 bins
    binwidth = fd.step[end] ÷ bin_n

    DataFrames.transform!(fd, :step => (x->(x .- 1) .÷ binwidth .* binwidth) => :stepbin)
    # barplot of patch bases
    barbase = Plots.bar(ps.base, orientation=:horizontal, legend=false, xlabel="base")
    # group level choice
    gfd = @linq fd |>
        groupby([:stepbin, :gid]) |>
	combine(patch_prop = proportion(:chosen_patch, patch_n))
    gfd.patch_id = repeat(ps.id, bin_n * forager_grp_n)
    for gid in 1:forager_grp_n
        z = Matrix(DataFrames.unstack(gfd[gfd.gid .== gid, :], :stepbin, :patch_id, :patch_prop))'
        x = z[1, :]
        z = z[2:end, :]
        push!(choice_plts, Plots.heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="group $gid patch choice"))
    end
    # individual patch choice
    ifd = @linq fd |>
        groupby([:stepbin, :id]) |>
        combine(patch_prop = proportion(:chosen_patch, patch_n))
    ifd.patch_id = repeat(ps.id, bin_n * forager_n)
    for fid in fs.id
        z = Matrix(DataFrames.unstack(ifd[ifd.id .== fid, :], :stepbin, :patch_id, :patch_prop))'
        x = z[1, :]
        z = z[2:end, :]
        push!(choice_plts, Plots.heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="forager $fid patch choice\nα=$(fs.α[fid-patch_n])\nβ=$(fs.β[fid-patch_n])\nρ=$(fs.ρ[fid-patch_n])"))
    end
    # individual Q
    for fid in fs.id
        Qfd = flatten(fd[fd.id .== fid, :], :Q)
        Qfd.patch_id = repeat(ps.id, fd.step[end])
        z = Matrix(DataFrames.unstack(Qfd, :step, :patch_id, :Q))'
        x = z[1, :]
        z = z[2:end, :]
        push!(Q_plts, Plots.heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="forager $fid Q"))
    end
    #MAYBE: color barplot with .series_list
    barbase, choice_plts, Q_plts
    #plot(choice_plts[3], barbase, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)
end
function test1()
    # one forager, ten patches
    model1 = init_modelA(;
        forager_ns=[1],
        μ_logαs=[log(0.1)], σ_logαs=[0.0],
        μ_logρs=[log(0.9)], σ_logρs=[0.0],
        μ_logβs=[log(0.72)], σ_logβs=[0.0],
        patch_n=10,
    )
    collect_modelA(model1)
end
function test2()
    # ten foragers, ten patches
    model1 = init_modelA(;
        forager_ns=[10],
        μ_logαs=[log(0.1)], σ_logαs=[0.3],
        μ_logρs=[log(0.9)], σ_logρs=[0.03],
        μ_logβs=[log(0.72)], σ_logβs=[0.8],
        patch_n=10,
    )
    collect_modelA(model1)
end
function test3()
    # two groups of 5 foragers with different σ_logα
    model1 = init_modelA(;
        σ_logαs=[0.2, 0.3],
        patch_n=10,
        forager_ns=[5, 5],
    )
    collect_modelA(model1)
end
function test4()
    # two groups of 5 foragers with different σ_logρ
    model1 = init_modelA(;
        σ_logρs=[0.01, 0.03],
        patch_n=10,
        forager_ns=[5, 5],
    )
    collect_modelA(model1)
end
function test5()
    # two groups of 5 foragers with different σ_logρ
    model1 = init_modelA(;
        σ_logβs=[0.6, 0.8],
        patch_n=10,
        forager_ns=[5, 5],
    )
    collect_modelA(model1)
end
