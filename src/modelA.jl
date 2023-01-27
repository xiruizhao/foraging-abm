mutable struct ForagerA <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int # group id
    α::Float64 # learning rate ∈(0, 1); α↑ => discount past experience
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
    μ_rew::Float64
    σ_rew::Float64
    rew_sampler::Distributions.Normal{Float64}
    shock_prob::Float64
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
            learning_store = [Float64[] for _ in 1:length(model.foragers)]
            for forager in patch.visited_by
                forager.rew = rand(patch.rew_sampler)
                push!(learning_store[forager.gid], forager.rew)
                U = sym_utility_risk(forager.rew, forager.ρ)
                forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
            end
            # in-group communication for all groups
            # if model.comm
            #     for (gid, rews) in enumerate(learning_store)
            #         if length(rews) > 0
            #             mean_rew = Statistics.mean(rews)
            #             for forager in model.foragers[gid]
            #                 if forager.chosen_patch != patch.id
            #                     U = sym_utility_risk(mean_rew, forager.ρ)
            #                     forager.Q[patch.id] += forager.α * (U - forager.Q[patch.id])
            #                 end
            #             end
            #         end
            #     end
            # end
            # enable communication for group 2
            if model.comm
                gid = 2
                rews = learning_store[gid]
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
        if rand() < patch.shock_prob
            patch.μ_rew = rand(model.μ_rew_sampler)
            patch.rew_sampler = Distributions.Normal(patch.μ_rew, patch.σ_rew)
        end
    end
end
function init_modelA(;
    forager_grp_n=2,
    forager_n=30,
    copy_α=true,
    μ_logα=log(0.1), σ_logα=0,
    copy_ρ=true,
    μ_logρ=log(1), σ_logρ=0, # disable risk preference
    copy_β=true,
    μ_logβ=log(5), σ_logβ=0,
    patch_n=50,
    μ_μ_rew=5,
    σ_rew_perc=0.3,
    shock_prob=0,
    comm=false,
)
    model = Agents.ABM(
               Union{ForagerA, PatchA},
               Agents.GridSpace((1,1));
               properties=Dict(
                               :patches=>Vector{PatchA}(undef, patch_n),
                               :foragers=>[Vector{ForagerA}(undef, forager_n) for _ in 1:forager_grp_n],
                               :comm=>comm,
                               :μ_rew_sampler=>Distributions.Poisson(μ_μ_rew),
                              ),
               warn=false
               )
    # add patches
    μ_rews = sort(rand(model.μ_rew_sampler, patch_n))
    σ_rews = μ_rews .* σ_rew_perc
    for i in 1:patch_n
        model.patches[i] = Agents.add_agent!((1,1), PatchA, model, μ_rews[i], σ_rews[i], Distributions.Normal(μ_rews[i], σ_rews[i]), shock_prob, ForagerA[])
    end
    # add foragers
    # αs = rand(Distributions.Uniform(0.07, 0.13), 1000)
    # ρs = ones(1000)
    # βs = rand(Distributions.Uniform(4, 6), 1000)
    #
    if copy_α
        αs = rand(Distributions.LogNormal(μ_logα, σ_logα), forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
    end
    if copy_ρ
        ρs = rand(Distributions.LogNormal(μ_logρ, σ_logρ), forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
    end
    if copy_β
        βs = rand(Distributions.LogNormal(μ_logβ, σ_logβ), forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
    end
    for gid in 1:forager_grp_n
        if !copy_α
            αs = rand(Distributions.LogNormal(μ_logα[gid], σ_logα[gid]), forager_n)# log(α) ∼ 𝒩 (μ_α, σ_α)
        end
        if !copy_ρ
            ρs = rand(Distributions.LogNormal(μ_logρ[gid], σ_logρ[gid]), forager_n) # log(ρ) ∼ 𝒩 (μ_logρ, σ_logρ)
        end
        if !copy_β
            βs = rand(Distributions.LogNormal(μ_logβ[gid], σ_logβ[gid]), forager_n) # log(β) ∼ 𝒩 (μ_logβ, σ_logβ)
        end
        for i in 1:forager_n
            model.foragers[gid][i] = Agents.add_agent!((1,1), ForagerA, model, gid, αs[i], ρs[i], βs[i], 0, 0.0, ones(patch_n))
        end
    end
    model
end
function collect_modelA(model; steps=3000)
    patch_static = DataFrames.DataFrame(id=Int[], μ_rew=Float64[], σ_rew=Float64[])
    patch_dynamic = DataFrames.DataFrame(id=Int[], step=Int[], μ_rew=Float64[])
    forager_static = DataFrames.DataFrame(id=Int[], gid=Int[], α=Float64[], ρ=Float64[], β=Float64[], U=Vector{Float64}[]) # U is the real utility of every patch
    forager_dynamic = DataFrames.DataFrame(step=Int[], id=Int[], gid=Int[], chosen_patch=Int[], rew=Float64[], Q=Vector{Float64}[])
    for patch in model.patches
        push!(patch_static, (patch.id, patch.μ_rew, patch.σ_rew))
    end
    for forager in vcat(model.foragers...)
        push!(forager_static, (forager.id, forager.gid, forager.α, forager.ρ, forager.β, (patch_static.μ_rew).^forager.ρ))
    end
    for step in 1:steps
        Agents.step!(model, Agents.dummystep, step_modelA!, 1)
        for patch in model.patches
            push!(patch_dynamic, (patch.id, step, patch.μ_rew))
        end
        for forager in vcat(model.foragers...)
            push!(forager_dynamic, (step, forager.id, forager.gid, forager.chosen_patch, forager.rew, copy(forager.Q)))
        end
    end
    patch_static, patch_dynamic, forager_static, forager_dynamic
end
function plot_modelA(ps, fs, fd)
    ps.id = map(string, ps.id)
    gcp = [] # group choice plots
    cp = [] # agent choice plots
    qp = [] # agent Q plots
    rew_plts = []
    forager_grp_n = fd.gid[end]
    forager_n = nrow(fs)
    patch_n = nrow(ps)
    bin_n = 200 # 200 bins
    binwidth = fd.step[end] ÷ bin_n

    DataFrames.transform!(fd, :step => (x->(x .- 1) .÷ binwidth .* binwidth) => :stepbin)
    # group level choice
    gfd = @linq fd |>
        groupby([:stepbin, :gid]) |>
	combine(patch_prop = proportion(:chosen_patch, patch_n))
    gfd.patch_id = repeat(ps.id, bin_n * forager_grp_n)
    for gid in 1:forager_grp_n
        z = Matrix(DataFrames.unstack(gfd[gfd.gid .== gid, :], :stepbin, :patch_id, :patch_prop))'
        x = z[1, :]
        z = z[2:end, :]
        tmp = Plots.heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="group $gid patch choice")
        tmp = Plots.scatter!(zeros(Int, patch_n), collect(1:patch_n).-0.5, markersize=ps.base, label="base")
        push!(choice_plts, tmp)
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
        push!(choice_plts, Plots.heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="forager $fid patch choice\nα=$(Formatting.sprintf1("%.3f", fs.α[fid-patch_n])), β=$(Formatting.sprintf1("%.3f", fs.β[fid-patch_n])), ρ=$(Formatting.sprintf1("%.3f", fs.ρ[fid-patch_n]))"))
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
    # reward
    rfd = @linq fd |>
        groupby([:stepbin, :gid]) |>
        combine(mean_rew = Statistics.mean(:rew))
    rew_plt = @df rfd plot(:stepbin, :mean_rew, group=:gid, title="mean reward")
    #
    αd = @df fs density(:α, group=:gid, title="α distribution")
    ρd = @df fs density(:ρ, group=:gid, title="ρ distribution")
    βd = @df fs density(:β, group=:gid, title="β distribution")
    choice_plts, Q_plts, rew_plt, αd, ρd, βd
    #plot(choice_plts[3], barbase, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)
end

function one_run()
    # constant
    nrun = 1000 # number of simulation runs per config
    steps = 3000
    patch_n = 50
    forager_grp_n = 1
    forager_n = 1
    ρ = 1
    μ_μ_rew = 5
    shock_prob = 0
    # variable
    βs = collect(0:0.5:15)
    nrow = length(βs)
    αs = collect(0.1:0.01:0.2)
    ncol = length(αs)
    perfs = [[Threads.Atomic() for _ in 1:nrow] for __ in 1:ncol]
    model = Agents.ABM( Union{ForagerA, PatchA}, Agents.GridSpace((1,1)); properties=Dict( :patches=>Vector{PatchA}(undef, patch_n), :foragers=>[Vector{ForagerA}(undef, ncol) for _ in 1:nrow], :μ_rew_sampler=>Distributions.Poisson(μ_μ_rew), :comm=>false), warn=false)
    # constant patches
    μ_rews = sort(rand(model.μ_rew_sampler, patch_n))
    σ_rews = μ_rews .* 0.0
    for i in 1:patch_n
        model.patches[i] = Agents.add_agent!((1,1), PatchA, model, μ_rews[i], σ_rews[i], Distributions.Normal(μ_rews[i], σ_rews[i]), shock_prob, ForagerA[])
    end
    runs = []
    for i in 1:nrow
        for j in 1:ncol
            model.foragers[i][j] = Agents.add_agent!((1,1), ForagerA, model, 1, αs[j], ρ, βs[i], 0, 0.0, ones(patch_n))
        end
    end
    _, _, fs, fd = collect_modelA(model)
    run = @linq fd |>
        groupby([:id]) |>
        combine(sumrew = sum(:rew))
    fs, run
end
