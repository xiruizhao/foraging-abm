### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
using Revise

# ╔═╡ 856e4422-7c2b-11eb-02b8-6d725cada156
using Gadfly

# ╔═╡ 0d17c998-7c92-11eb-2092-2be780a7fc91
import Pkg; Pkg.add("Revise")

# ╔═╡ 8b3ac81e-7c2b-11eb-3ea7-1b905970e72a
import Mod

# ╔═╡ 8d814bca-7c2b-11eb-0aea-e1562ee7a8c5
Mod.model_init()

# ╔═╡ 56593a08-7c92-11eb-0794-25748b73c11a
Mod.main()

# ╔═╡ 9c09af86-7c2b-11eb-2218-fdf87607746f
# plot(df[1], x=:step, y=:μ, color=:id, Geom.point, Geom.line)

# ╔═╡ e6a6dc20-ecd3-48cb-b1ec-71fea0938075
# draw(SVG("plot.svg"), p)

# ╔═╡ Cell order:
# ╠═0d17c998-7c92-11eb-2092-2be780a7fc91
# ╠═7d1c8c92-7c2b-11eb-206d-6d64d8f191fb
# ╠═856e4422-7c2b-11eb-02b8-6d725cada156
# ╠═8b3ac81e-7c2b-11eb-3ea7-1b905970e72a
# ╠═8d814bca-7c2b-11eb-0aea-e1562ee7a8c5
# ╠═56593a08-7c92-11eb-0794-25748b73c11a
# ╠═9c09af86-7c2b-11eb-2218-fdf87607746f
# ╠═e6a6dc20-ecd3-48cb-b1ec-71fea0938075
