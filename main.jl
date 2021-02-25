import Agents
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

mutable struct Food <: Agents.AbstractAgent
    id::Int
    pos::Dims{2}
    amount::Int
    growth_rate::Int
    Food(id, pos; amount=0, growth_rate=0) = new(id, pos, amount, growth_rate)
end
function agent_step!(food::Food, model::Agents.ABM)
    food.amount += food.growth_rate
end

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

model = model_init()
adata, _ = Agents.run!(model, agent_step!, 100; adata=[:amount])
adata

# function step
# e = model.space.extent
# for i in 0:100
#     i > 0 && agents.step!(model, agent_step!, 1)
#     p1 = agents.plotabm( model;)
#     title!(p1, "step $(i)")
# end
