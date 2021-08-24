using Turing
using Bijectors: ADBackend

struct SplitMerge{AD} <: ADBijector{AD, 0} end

(b::SplitMerge)(x) = split(x)
(ib::Inverse{<: SplitMerge})(y) = merge(y)

function split(input)
    splitw, splitμ, splitv, u1, u2, u3 = input
    w1 = splitw*u1
    w2 = splitw*(1-u1)
    μ1 = splitμ - u2*sqrt(splitv)*sqrt(w2/w1)
    μ2 = splitμ + u2*sqrt(splitv)*sqrt(w1/w2)
    v1 = u3*(1-u2^2)*splitv*(splitw/w1)
    v2 = (1-u3)*(1-u2^2)*splitv*(splitw/w2)
    return vcat(w1, w2, μ1, μ2, v1, v2)
end

function merge(input)
    w1, w2, μ1, μ2, v1, v2 = input
    neww = w1+w2
    newμ = (w1*μ1 + w2*μ2)/neww
    newv = (w1*(μ1^2+v1)+w2*(μ2^2+v2))/neww-newμ^2
    u1 = w1/(neww)
    u2 = (newμ-μ1)/(sqrt(newv*w2/w1))
    u3 = (v1*w1)/((1-u2^2)*newv*neww)
    return vcat(neww, newμ, newv, u1, u2, u3)
end

sm = SplitMerge{ADBackend()}()

x1 = [0.9776180916782476, -0.5708701548955284, 78.1934238492967, 0.7308482706433362, 0.5756331160188313, 0.7995627403039852]

x2 = [0.9776180916782476, 0.02238190832175231, -0.5708701548955284, -1.3206030300034364, 78.1934238492967, 1407.8698497409719]
