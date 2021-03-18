### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
using Revise

# ╔═╡ d6ba2b5c-80ea-11eb-07fe-51ab446766bb
using DataFrames

# ╔═╡ 483e172c-8636-11eb-28e0-fbc279a1ef09
using DataFramesMeta

# ╔═╡ b9cf65de-864b-11eb-02a7-a521105ea77f
using Statistics

# ╔═╡ 60273126-861f-11eb-3782-7b239664506e
using StatsPlots

# ╔═╡ 97800a90-809e-11eb-0158-3b2f7cdad4b0
import XiruiModels as XM

# ╔═╡ 97276fc0-809e-11eb-1a5a-2f8d06584d48
begin
	model = XM.init_model0(; patch_n=30, σ_walk=0.5, σ_rew=0.5, forager_ns=[5, 5], μ_logαs=[log(0.01), log(0.1)]);
	ps, fs, pd, fd = XM.collect_model0(model; steps=5e4);
	nothing
end

# ╔═╡ 3e0a007c-80dc-11eb-0212-cd13ee73c375
@df pd plot(:step, :μ_rew, group=:id, xlabel="step", ylabel="μ_rew", title="μ_rew random walk")

# ╔═╡ b94737c4-8637-11eb-2019-01634c4ed149
begin
	binwidth = 250 # steps
	transform!(fd, :step => (x->(x .- 1) .÷ binwidth .* binwidth) => :stepbin)
	gfd = @linq fd |> 
		groupby([:stepbin, :gid]) |> 
		combine(grew = mean(:rew))
	@df gfd plot(:stepbin, :grew, group=:gid, xlabel="step (binwidth = $binwidth)", ylabel="rew", title="group-level mean(rew) and individual rew")
	ifd = @linq fd |> 
		groupby([:stepbin, :gid, :id]) |> 
		combine(rew = mean(:rew))
	@df ifd plot!(:stepbin, :rew, group=:id, color=:gid, legend=false, alpha=0.2)
	top10 = combine(groupby(pd, :step), :μ_rew => (x->sum(sort(x, rev=true)[1:10])) => :top10)
	@df top10 plot!(:step, :top10./10)
end

# ╔═╡ a62b9dfe-8592-11eb-3a54-e72a8dbcc1be
#savefig(p, "fig.svg")

# ╔═╡ Cell order:
# ╠═7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
# ╠═d6ba2b5c-80ea-11eb-07fe-51ab446766bb
# ╠═483e172c-8636-11eb-28e0-fbc279a1ef09
# ╠═b9cf65de-864b-11eb-02a7-a521105ea77f
# ╠═60273126-861f-11eb-3782-7b239664506e
# ╠═97800a90-809e-11eb-0158-3b2f7cdad4b0
# ╠═97276fc0-809e-11eb-1a5a-2f8d06584d48
# ╠═3e0a007c-80dc-11eb-0212-cd13ee73c375
# ╠═b94737c4-8637-11eb-2019-01634c4ed149
# ╠═a62b9dfe-8592-11eb-3a54-e72a8dbcc1be
