### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ bceab76a-b999-11eb-27aa-e10cce4a889a
begin
	using Revise
	using StatsPlots
	using PlutoUI
	using DataFrames
	using DataFramesMeta
	import Agents
	import Distributions
	using Statistics
	import XiruiModels as XM
	import Random
	import Formatting
	seed = Int(time(Libc.strptime("%Y-%m-%dT%H:%M:%S", "2021-05-23T12:00:00")))
end;

# ╔═╡ 175388a2-e54a-4487-8df6-f86242042763
md"1. Diversity in learning rate"

# ╔═╡ 354ac196-b746-4870-b3cd-5676ffcb39ba
let forager_grp_n = 2, σ_logβ = 0.1
	# σ_logα = [0.05, 0.3]
	μ_logβ = [log(5), log(5)]
	Random.seed!(seed)
	μ_logα = [log(0.2), log(0.2)]
	model1 = XM.init_modelA(; forager_grp_n, copy_β=true, σ_logβ, comm=true)
	ps, pd, fs, fd = XM.collect_modelA(model1)
	# group choice plots
	gcps = []
	forager_n = 30
	patch_n = nrow(ps)
	bin_n = 200 # 200 bins
    binwidth = fd.step[end] ÷ bin_n
	transform!(fd, :step => (x->(x .- 1) .÷ binwidth .* binwidth) => :stepbin)
	gfd = @linq fd |>
        groupby([:stepbin, :gid]) |>
		combine(patch_prop = XM.proportion(:chosen_patch, patch_n), mean_rew = mean(:rew))
    gfd.patch_id = repeat(ps.id, bin_n * forager_grp_n)
    for gid in 1:forager_grp_n
        z = Matrix(unstack(gfd[gfd.gid .== gid, :], :stepbin, :patch_id, :patch_prop))'
        x = z[1, :]
        z = z[2:end, :]
        gcp = heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="group $gid patch choice")
        #gcp = scatter!(zeros(Int, patch_n), collect(1:patch_n).-0.5, markersize=ps.μ_rew, label="μ_rew")
        push!(gcps, gcp)
    end
	# μ_rew
	z = Matrix(unstack(pd, :step, :id, :μ_rew))'
	x = z[1, :]
	z = z[2:end, :]
	μ_rewp = heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="patch μ_rew")
	# reward
    rp = @df gfd plot(:stepbin, :mean_rew, group=:gid, xlabel="step", title="group mean reward")
	# individual choice
	cps = []
	ifd = @linq fd |>
        groupby([:stepbin, :id]) |>
        combine(patch_prop = XM.proportion(:chosen_patch, patch_n), mean_rew = mean(:rew))
	ifd.patch_id = repeat(ps.id, bin_n * forager_n * forager_grp_n)
    for fid in fs.id
        z = Matrix(unstack(ifd[ifd.id .== fid, :], :stepbin, :patch_id, :patch_prop))'
        x = z[1, :]
        z = z[2:end, :]
        push!(cps, heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="forager $(fid-patch_n) patch choice\nα=$(Formatting.sprintf1("%.3f", fs.α[fid-patch_n])), β=$(Formatting.sprintf1("%.3f", fs.β[fid-patch_n])), ρ=$(Formatting.sprintf1("%.3f", fs.ρ[fid-patch_n]))"))
    end
    # individual Q
	qps = []
    for fid in fs.id
        Qfd = flatten(fd[fd.id .== fid, :], :Q)
        Qfd.patch_id = repeat(ps.id, fd.step[end])
        z = Matrix(unstack(Qfd, :step, :patch_id, :Q))'
        x = z[1, :]
        z = z[2:end, :]
        push!(qps, heatmap(x, ps.id, z, xlabel="step", ylabel="patch_id", title="forager $(fid-patch_n) Q"))
    end
	# individual reward
	rps = []
	for fid in fs.id
        push!(rps, @df ifd[ifd.id .== fid, :] plot(:stepbin, :mean_rew, xlabel="step", title="forager $fid mean_rew"))
    end
	# parameters
	αd = @df fs density(:α, group=:gid, xlims=[0, Inf], title="α distribution")
    βd = @df fs density(:β, group=:gid, title="β distribution")
	
	d1 = plot(gcps[1], gcps[2], rp, βd, layout=grid(2, 2), size=(800, 500))
	# cps[argmin(fs.β)], qps[argmin(fs.β)], cps[argmax(fs.β)], qps[argmax(fs.β)],
	savefig(d1, "d4.png")
	d1
end

# ╔═╡ 14471616-1f8b-4e1e-8a8c-3e4376b4086e
#=
let patch_n = 50, forager_grp_n = 1, forager_n = 1, α = 0.1, ρ = 1, μ_μ_rew = 5, shock_prob = 0
	Random.seed!(seed)
	βs = collect(0:0.01:10)
	forager_pool_n = length(βs)
	perfs = Vector{Float64}(undef, forager_pool_n)
	model = Agents.ABM(
               Union{XM.ForagerA, XM.PatchA},
               Agents.GridSpace((1,1));
               properties=Dict(
                               :patches=>Vector{XM.PatchA}(undef, patch_n),
                               :foragers=>[Vector{XM.ForagerA}(undef, forager_n) for _ in 1:forager_grp_n], 
							   :μ_rew_sampler=>Distributions.Poisson(μ_μ_rew),
							    :comm=>false
                              ),
               warn=false
               )
	# add patches
	μ_rews = sort(rand(model.μ_rew_sampler, patch_n))
    σ_rews = μ_rews .* 0.3
    for i in 1:patch_n
        model.patches[i] = Agents.add_agent!((1,1), XM.PatchA, model, μ_rews[i], σ_rews[i], Distributions.Normal(μ_rews[i], σ_rews[i]), shock_prob, XM.ForagerA[])
    end
    for i in 1:forager_pool_n
	    model.foragers[1][1] = Agents.add_agent!((1,1), XM.ForagerA, model, 1, α, ρ, βs[i], 0, 0.0, ones(patch_n))
		_, _, _, fd = XM.collect_modelA(model)
		perfs[i] = sum(fd.rew)
		Agents.kill_agent!(model.foragers[1][1], model)
	end
	plot(perfs)
end
=#

# ╔═╡ e7bd7fee-c05a-45bb-8b20-d146545ebb16
begin
	runs = repeat(Any[0], 10)
	Threads.@threads for i in 1:10
		runs[i] =  XM.one_run()
	end
	fs, fd = runs[1]
	fs = @linq fs |>
		select(:id, :α, :β)
	fs.sumrew = fd.sumrew
	z = Matrix(unstack(fs, :β, :α, :sumrew))'
    x = z[1, :]
    z = z[2:end, :]
    heatmap(0:0.5:15, 0.1:0.01:0.2, z, xlabel="β", ylabel="α", title="total reward")
end

# ╔═╡ Cell order:
# ╠═bceab76a-b999-11eb-27aa-e10cce4a889a
# ╟─175388a2-e54a-4487-8df6-f86242042763
# ╠═354ac196-b746-4870-b3cd-5676ffcb39ba
# ╠═14471616-1f8b-4e1e-8a8c-3e4376b4086e
# ╠═e7bd7fee-c05a-45bb-8b20-d146545ebb16
