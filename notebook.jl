### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ a31a5b04-7454-11eb-194b-d1ac68cb0b79
import Agents

# ╔═╡ 36e05312-7710-11eb-0851-b900b2ca4890
# import LinearAlgebra as LinAlg
# import Plots
# import Random

mutable struct Forager <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    gid::Int # group id
    vision::Int
    # impulsivity::Float64
    mrate::Int # metabolic rate
    age::Int
    max_age::Int
    Forager(id, pos; gid=0, vision=0, mrate=0, age=0, max_age=0) = new(id, pos, gid, vision, mrate, age, max_age)
end

# ╔═╡ 370e795e-7710-11eb-17d6-5b0bccffd1e4
mutable struct Food <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    amount::Int
    growth_rate::Int
    Food(id, pos; amount=0, growth_rate=0) = new(id, pos, amount, growth_rate)
end

# ╔═╡ 370df312-7710-11eb-00cb-3dfe4eebe2bf
function agent_step!(forager::Forager, model::Agents.ABM)
    neighbors = collect(Agents.nearby_agents(forager, model, forager.vision))
    for neighbor in neighbors
	# food_max = maximum(x->x.amount, filter(x->x isa Food, neighbors))
	if neighbor isa Food
	    neighbor.amount -= forager.mrate
	end
    end
    forager.age += 1
    if forager.age >= forager.max_age
	Agents.kill_agent!(forager, model)
    else
	Agents.move_agent!(forager, model)
    end
end

# ╔═╡ 371cc982-7710-11eb-38bf-93bf1efbfb69
function agent_step!(food::Food, model::Agents.ABM)
    food.amount += food.growth_rate
end

# ╔═╡ 371d68ba-7710-11eb-2204-bf4cc20134d8
function model_init()
    model = Agents.ABM(Union{Forager, Food}, Agents.GridSpace((100, 100), periodic = true, metric = :euclidean), scheduler = Agents.random_activation)
    for _ in 1:10
	Agents.add_agent!(Forager, model; gid=1, vision=5, mrate=1, age=0, max_age=100)
	Agents.add_agent!(Forager, model; gid=2, vision=7, mrate=2, age=0, max_age=100)
    end
    for _ in 1:20
	Agents.add_agent!(Food, model; amount=5, growth_rate=1)
    end
    return model
end

# ╔═╡ 3729b804-7710-11eb-31ff-c12c0b71e5f3
model = model_init()

# ╔═╡ 372a5c82-7710-11eb-0e06-0d4fc1312536
adata, _ = Agents.run!(model, agent_step!, 100; adata=[:amount]);

# ╔═╡ 373775a2-7710-11eb-0473-01d21ff5739e
adata

# ╔═╡ 37384d74-7710-11eb-12de-abfddd1ee82d
# function step
# e = model.space.extent
# for i in 0:100
#     i > 0 && agents.step!(model, agent_step!, 1)
#     p1 = agents.plotabm( model;)
#     title!(p1, "step $(i)")
# end

# ╔═╡ Cell order:
# ╠═a31a5b04-7454-11eb-194b-d1ac68cb0b79
# ╠═36e05312-7710-11eb-0851-b900b2ca4890
# ╠═370df312-7710-11eb-00cb-3dfe4eebe2bf
# ╠═370e795e-7710-11eb-17d6-5b0bccffd1e4
# ╠═371cc982-7710-11eb-38bf-93bf1efbfb69
# ╠═371d68ba-7710-11eb-2204-bf4cc20134d8
# ╠═3729b804-7710-11eb-31ff-c12c0b71e5f3
# ╠═372a5c82-7710-11eb-0e06-0d4fc1312536
# ╠═373775a2-7710-11eb-0473-01d21ff5739e
# ╠═37384d74-7710-11eb-12de-abfddd1ee82d
