### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
using Revise

# ╔═╡ eb4f215e-80d7-11eb-1968-191861ef0cdb
using StatsPlots

# ╔═╡ d6ba2b5c-80ea-11eb-07fe-51ab446766bb
using DataFrames

# ╔═╡ 97800a90-809e-11eb-0158-3b2f7cdad4b0
import XiruiModels as XM

# ╔═╡ 97276fc0-809e-11eb-1a5a-2f8d06584d48
model = XM.init_model0(; patch_n=3, σ_walk=0.1, σ_rew=0.0, forager_n=10)

# ╔═╡ d89b4e88-809e-11eb-3f02-47028239f109
ps, fs, pd, fd = XM.collect_model0(model; steps=1e4);

# ╔═╡ 3e0a007c-80dc-11eb-0212-cd13ee73c375
p = plot(Matrix(XM.DataFrames.DataFrame(fd.Q[1:20:end]))', xlabel="step", ylabel="Q", legend=false, title="learning walking and variable patch, α=$(fs.α[1])")

# ╔═╡ fa575dc2-820f-11eb-2a93-890964ae7706
q = plot(Matrix(XM.DataFrames.DataFrame(fd.target_Q[1:20:end]))', xlabel="step", ylabel="target Q", legend=false, title="learning walking and variable patch, α=$(fs.α[1])")

# ╔═╡ e6a6dc20-ecd3-48cb-b1ec-71fea0938075
#savefig(p, "fig.svg")

# ╔═╡ Cell order:
# ╠═7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
# ╠═eb4f215e-80d7-11eb-1968-191861ef0cdb
# ╠═d6ba2b5c-80ea-11eb-07fe-51ab446766bb
# ╠═97800a90-809e-11eb-0158-3b2f7cdad4b0
# ╠═97276fc0-809e-11eb-1a5a-2f8d06584d48
# ╠═d89b4e88-809e-11eb-3f02-47028239f109
# ╠═3e0a007c-80dc-11eb-0212-cd13ee73c375
# ╠═fa575dc2-820f-11eb-2a93-890964ae7706
# ╠═e6a6dc20-ecd3-48cb-b1ec-71fea0938075
