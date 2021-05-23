function argmaxes(x::Vector)
    max_elem = x[1]
    ret = [1]
    for (i, elem) in Iterators.drop(enumerate(x), 1)
        if elem == max_elem
            push!(ret, i)
        elseif elem > max_elem
            max_elem = elem
            empty!(ret)
            push!(ret, i)
        end
    end
    ret
end
function proportion(patch_ids, patch_n)
    freq = zeros(Int, patch_n)
    for patch_id in patch_ids
        freq[patch_id] += 1
    end
    freq ./ length(patch_ids)
end
function edis(a, b)
    (a .- b) .^ 2 |> sum |> sqrt
end
function entropy(x::Vector{T}) where T
    freq = Dict{T, Int}()
    for elem in x
        freq[elem] = get(freq, elem, 0) + 1
    end
    prob = values(freq) ./ length(x)
    -sum(prob .* log.(2, prob))
end
function mn_softmax_sample(x, β) # maximum-normalized
    # y = copy(x)
    y = x ./ maximum(x)
    y .= exp.(β .* y)
    y ./= sum(y)
    StatsBase.sample(StatsBase.pweights(y))
end
function sym_utility_risk(V, ρ)
    if V < 0.0
        -sym_utility_risk(-V, ρ)
    else
        V^ρ
    end
end
