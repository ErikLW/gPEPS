# gPEPS.jl
`gPEPS` is a Julia package for optimizing PEPS (Projected Entangled Pair States) using the simple update. 
It is build on the [TensorKit-library](https://github.com/QuantumKitHub/TensorKit.jl) (v0.16.3) and
implements the simple update as formulated in this [paper](https://arxiv.org/abs/1808.00680)
# Feature Overview
- **PEPS Framework**: The simple update optimization can be performed on arbitrary periodic lattices in arbitrary dimensions.
- **Expectation values**: Expectation values can be optained within the SU-framework.

# Minimal Example: Heisenberg Model on the square lattice in 2d

```julia

using gPEPS
using LinearAlgebra
using TensorKit

n = 2 #This sets the bond dimension of our PEPS

#This initializes the tensors we aim to optimize using the simple-update algorithm.
#We define two ingoing and two outgoing virtual indices and a physical index with dimension d = 2.
#Convention: For all tensors we define in this package, it is important, that the physical index is the last one!
A = rand(Float64, ℂ^n ⊗ ℂ^n, ℂ^n ⊗ ℂ^n ⊗ (ℂ^2)') 
B = rand(Float64, ℂ^n ⊗ ℂ^n, ℂ^n ⊗ ℂ^n ⊗ (ℂ^2)')
tens_arr = [A,B]

#We initialize the lambda-matrices. We have 4 unique edges in this setup, hence we define 4 lambda matrices.
#The lambda matrices are initialized as identity matrices.
lamb = isomorphism(ℂ^n,ℂ^n)
lamb_arr = [lamb for i in 1:4]

#This defines the structure matrix. 
#In this case we have two unique tensors (and hence two rows) and four unique edges (and hence four columns).
#The entry S[i,j] labels the index of tensor i, that is involved in edge j. Negative entries correspond to outgoing indices. 
str_mat = [-2 3 4 -1; 4 -1 -2 3] 

#To define the Hamiltonian we create spin-matrices and a 2-body Heisenberg_term (SᵢˣSⱼˣ+SᵢʸSⱼʸ+SᵢᶻSⱼᶻ = 1/2(Sᵢ⁺Sⱼ⁻ + Sᵢ⁻Sⱼ⁺)+SᵢᶻSⱼᶻ
S_x, S_y, S_z, Heisenberg_term, Id_term = create_spin_operators_SU(space_type = ℂ)

#We need to define a Hamiltonian term for unique edge of our system
ham_arr = [Heisenberg_term for i in 1:4]

#=This function takes:
    a) an array of input tensors
    b) an array of input lambda-tensors for the edges
    c) an array of Hamiltonian terms for every edge
    d) a structure matrix defining the lattice and its connectivity
    e) a dimension to which the virtual indices are truncated during the optimization

    It outputs an array of optimized local tensors and an array of optimized lambda tensors.
=#
tens_arr_opt, lamb_arr_opt = simple_update_optimization(tens_arr, lamb_arr, ham_arr, str_mat, n)

#evaluating the Hamiltonian terms on their corresponding lattices and deviding the sum by the number of unique sites gives the energy per site.
exp_val = 0
for i in 1:4
    #This function takes as a last input the number of the edge that we want to calculate an expecation value on.
    global exp_val += expectation_value_two_body(tens_arr_opt, lamb_arr_opt, str_mat, ham_arr[i], i)
end
exp_val_per_site = exp_val / 2

@info "the energy per site for the Heisenberg model in 2d on the square lattice in SU-approximation with bond-dim $n is e = $exp_val_per_site."

#these function calculate the expectation values of the spin-operators on the two different unique sites (last input variable).
m_x = 0
m_x += expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_x, 1)
m_x -= expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_x, 2)

m_y = 0
m_y += expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_y, 1)
m_y -= expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_y, 2)

m_z = 0
m_z += expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_z, 1)
m_z -= expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat, S_z, 2)

m = sqrt((m_x/2)^2 + (m_y/2)^2 + (m_z/2)^2)

@info "The staggered magnetization of the 2d Heisenberg model at bond-dimension d = $n is m = $(m) within the SU-approximation."
```