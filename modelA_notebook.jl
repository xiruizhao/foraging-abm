### A Pluto.jl notebook ###
# v0.14.4

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

# ╔═╡ fb85598c-bf7d-49ed-8e3d-2eaea586252a
md"# I. 10 patches, 1 forager"

# ╔═╡ 97276fc0-809e-11eb-1a5a-2f8d06584d48
begin
	patch_n = 10;
	ps, fs, fd, model = XM.test1();
	barbase, cp, qp = XM.modelA_heatmaps(ps, fs, fd); 
	plot(cp[2], barbase, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)
end

# ╔═╡ 7facd9ae-a32b-4f70-9d0b-28de3e38539b
plot(qp[1], barbase, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)

# ╔═╡ bf0b4179-5fee-47d8-a2c3-597e44bde9a1
md"## Change patch base"

# ╔═╡ 2d9cd6fc-1fd5-419a-ad44-fbb4ee240a2f
begin
	ps2, fs2, fd2, _ = XM.shock_modelA(model;);
	barbase2, cp2, qp2 = XM.modelA_heatmaps(ps2, fs2, fd2); 
	plot(cp2[2], barbase2, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)
end

# ╔═╡ 4618c3e0-9137-4457-a5c3-58c5caa2ce5b
plot(qp2[1], barbase2, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)

# ╔═╡ a62b9dfe-8592-11eb-3a54-e72a8dbcc1be
#@df gfd plot(:stepbin, :grew, group=:gid, xlabel="step (binwidth = $binwidth)", ylabel="rew", title="vicarious $vicarious. \ntop patches mean(rew), group-level mean(rew), and individual rew")
	#ifd = @linq fd |> 
	#	groupby([:stepbin, :gid, :id]) |> 
	#	combine(rew = mean(:rew))
	#@df ifd plot!(:stepbin, :rew, group=:id, color=:gid, legend=false, alpha=0.2)
	#top10 = combine(groupby(pd, :step), :μ_rew => (x->sum(sort(x, rev=true)[1:10])) => :top10)
	#p = @df top10 plot!(:step, :top10./10, dpi=400, fmt=:png)
	#savefig(p, "fig.png", fmt=:png)

	#ifd2 = @linq fd |> groupby([:stepbin, :gid, :id]) |> combine(entropy=XM.entropy(:chosen_patch));
	#@df ifd2 plot(:stepbin, :entropy, group=:id, color=:gid, legend=false, alpha=0.5)
#savefig(p, "fig.svg")

# ╔═╡ b02fb4db-2be4-4bb2-a339-619a55fd2bb6
md"# II. 10 patches, 10 foragers"

# ╔═╡ 16870639-fbe0-476e-88ae-776d4bafe42c
begin
	ps3, fs3, fd3, model3 = XM.test2();
	barbase3, cp3, qp3 = XM.modelA_heatmaps(ps3, fs3, fd3); 
	plot(cp3[1], barbase3, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)
end

# ╔═╡ b431c265-2120-475d-8cea-e34e16fa9a19
md"# III 10 patches, 2 groups of 5 foragers
"

# ╔═╡ 268e4e6d-a4c8-4e02-862e-a8c7989558d3
begin
	ps4, fs4, fd4, model4 = XM.test3();
	barbase4, cp4, qp4 = XM.modelA_heatmaps(ps4, fs4, fd4);
	md"## group1 σ_logα=0.2, group2 0.3"
end

# ╔═╡ b4a023d0-8a43-44bd-95e8-a49788b05ec7
plot(cp4[1], barbase4, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)

# ╔═╡ b8838616-21c4-4ef1-9e47-d20d6f77f6ce
plot(cp4[2], barbase4, layout=grid(1, 2, widths=[0.9, 0.1]), link=:y)

# ╔═╡ f3afc090-0af6-4b79-81a1-abae99b0118d
begin
	ps5, fs5, fd5, model5 = XM.test4();
	barbase5, cp5, qp5 = XM.modelA_heatmaps(ps5, fs5, fd5);
	md"## group1 σ_logρ=0.01 group2 0.03"
end

# ╔═╡ 2107cea0-00d5-4043-96d4-b06caa9b186c
plot(cp5[1], barbase5, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)

# ╔═╡ 85d8cc32-34a3-4037-8ac9-c3bc8346b97c
plot(cp5[2], barbase5, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)

# ╔═╡ 5ce1d92d-aa97-443d-91c6-2681c406ee67
begin
	ps6, fs6, fd6, model6 = XM.test5();
	barbase6, cp6, qp6 = XM.modelA_heatmaps(ps6, fs6, fd6);
	md"## group1 σ_logβ=0.6 group2 0.8"
end

# ╔═╡ 111554b8-1e8e-4935-a81f-1270c5399cb0
plot(cp6[1], barbase6, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)

# ╔═╡ 36361f05-8646-4af5-9b20-b02b072cd637
plot(cp6[2], barbase6, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)

# ╔═╡ 98ecb7c3-c253-44bb-b1b8-7300eb84d87d
md"# IV Change patch bases of two forager groups of different α
group1 σ_logα=0.2, group2 0.3
"

# ╔═╡ 0d647607-3be1-4235-b26c-6dee58cd2fe2
begin
	ps7, fs7, fd7, model7 = XM.test3();
	barbase7, cp7, qp7 = XM.modelA_heatmaps(ps7, fs7, fd7);
	plot(cp7[1], barbase7, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)
end

# ╔═╡ 7f2056f0-11e4-45af-8cfd-3bad6781c4cb
begin
	ps8, fs8, fd8, _ = XM.shock_modelA(model7)
	barbase8, cp8, qp8 = XM.modelA_heatmaps(ps8, fs8, fd8);
	plot(cp8[1], barbase8, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)
end

# ╔═╡ e4fe4bdb-d50b-4a26-8d28-5322f305de7f
begin
	plot(cp7[2], barbase7, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)
end

# ╔═╡ 55d02f1a-6001-47cd-8eca-c785c85689ca
begin
	plot(cp8[2], barbase8, layout=grid(1, 2, widths=[0.9, 0.1]),link=:y)
end

# ╔═╡ Cell order:
# ╠═7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
# ╠═d6ba2b5c-80ea-11eb-07fe-51ab446766bb
# ╠═483e172c-8636-11eb-28e0-fbc279a1ef09
# ╠═b9cf65de-864b-11eb-02a7-a521105ea77f
# ╠═60273126-861f-11eb-3782-7b239664506e
# ╠═97800a90-809e-11eb-0158-3b2f7cdad4b0
# ╠═fb85598c-bf7d-49ed-8e3d-2eaea586252a
# ╠═97276fc0-809e-11eb-1a5a-2f8d06584d48
# ╠═7facd9ae-a32b-4f70-9d0b-28de3e38539b
# ╠═bf0b4179-5fee-47d8-a2c3-597e44bde9a1
# ╠═2d9cd6fc-1fd5-419a-ad44-fbb4ee240a2f
# ╠═4618c3e0-9137-4457-a5c3-58c5caa2ce5b
# ╠═a62b9dfe-8592-11eb-3a54-e72a8dbcc1be
# ╠═b02fb4db-2be4-4bb2-a339-619a55fd2bb6
# ╠═16870639-fbe0-476e-88ae-776d4bafe42c
# ╠═b431c265-2120-475d-8cea-e34e16fa9a19
# ╠═268e4e6d-a4c8-4e02-862e-a8c7989558d3
# ╠═b4a023d0-8a43-44bd-95e8-a49788b05ec7
# ╠═b8838616-21c4-4ef1-9e47-d20d6f77f6ce
# ╠═f3afc090-0af6-4b79-81a1-abae99b0118d
# ╠═2107cea0-00d5-4043-96d4-b06caa9b186c
# ╠═85d8cc32-34a3-4037-8ac9-c3bc8346b97c
# ╠═5ce1d92d-aa97-443d-91c6-2681c406ee67
# ╠═111554b8-1e8e-4935-a81f-1270c5399cb0
# ╠═36361f05-8646-4af5-9b20-b02b072cd637
# ╠═98ecb7c3-c253-44bb-b1b8-7300eb84d87d
# ╠═0d647607-3be1-4235-b26c-6dee58cd2fe2
# ╠═7f2056f0-11e4-45af-8cfd-3bad6781c4cb
# ╠═e4fe4bdb-d50b-4a26-8d28-5322f305de7f
# ╠═55d02f1a-6001-47cd-8eca-c785c85689ca
