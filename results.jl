### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ bceab76a-b999-11eb-27aa-e10cce4a889a
begin
	using Revise
	using Plots
	import XiruiModels as XM
	import Random
	Random.seed!(20210521)
end;

# ╔═╡ 175388a2-e54a-4487-8df6-f86242042763
md"1. Diversity in learning rate"

# ╔═╡ 354ac196-b746-4870-b3cd-5676ffcb39ba
let σ_logαs = [0.2, 0.6]
	model1 = XM.init_modelA(; σ_logαs)
	ps, fs, fd, _ = XM.collect_modelA(model1)
	cp, qp, rp, αp, ρp, βp = XM.plot_modelA(ps, fs, fd)
	savefig(plot(cp[1], cp[2], rp, cp[3], qp[1], αp, ρp, βp, layout=grid(4, 2), size=(1000, 1414)), "1.png")
end

# ╔═╡ 82ae248a-3adc-487c-891f-36b7612b9a76
md"""2. Group 2 enabled in-group communication about the quality of patches
"""

# ╔═╡ 9d676531-fabe-47a9-99a2-cf08d1dc6f43
let comm = true
	model1 = XM.init_modelA(; comm)
	ps, fs, fd, _ = XM.collect_modelA(model1)
	cp, qp, rp, αp, ρp, βp = XM.plot_modelA(ps, fs, fd)
	savefig(plot(cp[1], cp[2], rp, αp, ρp, βp, layout=grid(3, 2), size=(1000, 1000)), "2.png")
end

# ╔═╡ c583e41e-87d5-4f76-9df0-645aa9a47202
md"3. Diversity in softmax temperature"

# ╔═╡ 7ded43b9-2e2c-4d33-816f-6f1148c4845b
let σ_logβs = [1, 2]
	model1 = XM.init_modelA(; σ_logβs, comm=false)
	ps, fs, fd, _ = XM.collect_modelA(model1)
	cp, qp, rp, αp, ρp, βp = XM.plot_modelA(ps, fs, fd)
	savefig(plot(cp[1], cp[2], rp, αp, ρp, βp, layout=grid(3, 2), size=(1000, 1000)), "3.png")
end

# ╔═╡ 5b763c7c-1991-4946-852e-94d7ff3b2418
let σ_logβs = [2, 4]
	model1 = XM.init_modelA(; σ_logβs, comm=false)
	ps, fs, fd, _ = XM.collect_modelA(model1)
	cp, qp, rp, αp, ρp, βp = XM.plot_modelA(ps, fs, fd)
	savefig(plot(cp[1], cp[2], rp, αp, ρp, βp, layout=grid(3, 2), size=(1000, 1000)), "4.png")
end

# ╔═╡ a46e88da-627f-47b2-ad67-9395f711bd5a
let σ_logβs = [2, 4]
	model1 = XM.init_modelA(; σ_logβs, comm=true)
	ps, fs, fd, _ = XM.collect_modelA(model1)
	cp, qp, rp, αp, ρp, βp = XM.plot_modelA(ps, fs, fd)
	savefig(plot(cp[1], cp[2], rp, αp, ρp, βp, layout=grid(3, 2), size=(1000, 1000)), "5.png")
end

# ╔═╡ Cell order:
# ╠═bceab76a-b999-11eb-27aa-e10cce4a889a
# ╟─175388a2-e54a-4487-8df6-f86242042763
# ╠═354ac196-b746-4870-b3cd-5676ffcb39ba
# ╟─82ae248a-3adc-487c-891f-36b7612b9a76
# ╠═9d676531-fabe-47a9-99a2-cf08d1dc6f43
# ╟─c583e41e-87d5-4f76-9df0-645aa9a47202
# ╠═7ded43b9-2e2c-4d33-816f-6f1148c4845b
# ╠═5b763c7c-1991-4946-852e-94d7ff3b2418
# ╠═a46e88da-627f-47b2-ad67-9395f711bd5a
