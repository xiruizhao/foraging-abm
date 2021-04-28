### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 4c588f92-80d5-11eb-3a6e-a3b2237e0439
using PlutoUI

# ╔═╡ 53ed8726-80d5-11eb-3a91-37f694d92700
using StatsPlots

# ╔═╡ 576969ba-80d5-11eb-33aa-254a6a258a5a
using Distributions

# ╔═╡ 69b616cc-80d5-11eb-2f08-fd7821c44d7b
md"""
# Tweaking parameters
"""

# ╔═╡ 5c3e8ac4-80d5-11eb-0021-9738f9064f6f
md"""
## Lognormal Distribution

μ, σ refer to the orginal normal distribution.

exp(μ) start $(@bind exp_μ_start TextField(; default="0.1")) step $(@bind exp_μ_step TextField(; default="0.01")) stop $(@bind exp_μ_stop TextField(; default="1"))

σ start $(@bind σ_start TextField(; default="0.1")) step $(@bind σ_step TextField(; default="0.01")) stop $(@bind σ_stop TextField(; default="1"))
"""

# ╔═╡ 68239c94-80d5-11eb-3c01-1bb0a43d1eea
md"""
exp(μ) $(join([exp_μ_start, exp_μ_step, exp_μ_stop], ":")) $(@bind exp_μ Slider(parse(Float64, exp_μ_start):parse(Float64, exp_μ_step):parse(Float64, exp_μ_stop)))

σ $(join([σ_start, σ_step, σ_stop], ":")) $(@bind σ Slider(parse(Float64, σ_start):parse(Float64, σ_step):parse(Float64, σ_stop)))
"""

# ╔═╡ 79796e9c-80d5-11eb-209f-2158b83171b2
histogram(rand(Distributions.LogNormal(log(exp_μ), σ), 10000), legend=false, title="exp(μ) = $exp_μ, σ = $σ")

# ╔═╡ 44e71d4c-8165-11eb-21f2-f18eb8d67d0f
# plot(Distributions.LogNormal(log(exp_μ), σ))

# ╔═╡ 7f04d8ee-80d5-11eb-09be-f330ed74e211
md"""
## Exponential Distribution

θ is the mean.

θ start $(@bind θ_start TextField(; default="0.1")) step $(@bind θ_step TextField(; default="0.01")) stop $(@bind θ_stop TextField(; default="1"))
"""

# ╔═╡ 80b665e8-80d5-11eb-28ae-7571ecbf9e54
md"""
θ $(join([θ_start, θ_step, θ_stop], ":")) $(@bind θ Slider(parse(Float64, θ_start):parse(Float64, θ_step):parse(Float64, θ_stop)))
"""

# ╔═╡ 8f19e326-80d5-11eb-2fb6-5dca8c275131
histogram(rand(Distributions.Exponential(θ), 10000), title="θ = $θ")

# ╔═╡ 417bf1d2-8165-11eb-3c1c-195e27752533
# plot(Distributions.Exponential(θ))

# ╔═╡ Cell order:
# ╠═4c588f92-80d5-11eb-3a6e-a3b2237e0439
# ╠═53ed8726-80d5-11eb-3a91-37f694d92700
# ╠═576969ba-80d5-11eb-33aa-254a6a258a5a
# ╟─69b616cc-80d5-11eb-2f08-fd7821c44d7b
# ╟─5c3e8ac4-80d5-11eb-0021-9738f9064f6f
# ╟─68239c94-80d5-11eb-3c01-1bb0a43d1eea
# ╠═79796e9c-80d5-11eb-209f-2158b83171b2
# ╠═44e71d4c-8165-11eb-21f2-f18eb8d67d0f
# ╟─7f04d8ee-80d5-11eb-09be-f330ed74e211
# ╟─80b665e8-80d5-11eb-28ae-7571ecbf9e54
# ╠═8f19e326-80d5-11eb-2fb6-5dca8c275131
# ╠═417bf1d2-8165-11eb-3c1c-195e27752533
