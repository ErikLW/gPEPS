using gPEPS
using TensorKit

#=
Here we take the example of the star lattice from the gPEPS paper (https://arxiv.org/abs/1808.00680).
Note that the structure matrix is shifted by one, since in our convention the physical index is in the last place.
Further we added signs to account for ingoing and outgoing indices, as is explained in the above paper.
This example can be used to reproduce the energies of Fig. 4 of the gPEPS paper.
=# 


str_mat_star = [-1 -2  3  0  0  0  0  0  0;
                1   0  0 -2 -3  0  0  0  0;
                0   1  0  2  0 -3  0  0  0;
                0   0  0  0  0  1 -2 -3  0;
                0   0  0  0  1  0  2  0 -3;
                0   0 -1  0  0  0  0  2  3]

n = 2

A1_star = rand(Float64, ℂ^n ⊗ ℂ^n ⊗ (ℂ^n)' ⊗ ℂ^2, one(ComplexSpace))
A2_star = rand(Float64, (ℂ^n)' ⊗ ℂ^n ⊗ ℂ^n ⊗ ℂ^2, one(ComplexSpace))
A3_star = rand(Float64, (ℂ^n)'⊗ (ℂ^n)' ⊗ ℂ^n ⊗ ℂ^2, one(ComplexSpace))
A4_star = rand(Float64, (ℂ^n)' ⊗ ℂ^n ⊗ ℂ^n ⊗ ℂ^2, one(ComplexSpace))
A5_star = rand(Float64, (ℂ^n)'⊗ (ℂ^n)' ⊗ ℂ^n ⊗ ℂ^2, one(ComplexSpace))
A6_star = rand(Float64, ℂ^n⊗ (ℂ^n)' ⊗ (ℂ^n)' ⊗ ℂ^2, one(ComplexSpace))

tens_arr_star = [A1_star, A2_star, A3_star, A4_star, A5_star, A6_star]

lamb = isomorphism(ℂ^n,ℂ^n)
lamb_arr_star = [lamb for i in 1:9]

S_x, S_y, S_z, Heisenberg_term, Id_term = gPEPS.create_spin_operators_SU(space_type = ℂ)

J_t = 0.05
ham_arr_star = [J_t*Heisenberg_term, J_t*Heisenberg_term,Heisenberg_term,
                 J_t*Heisenberg_term, Heisenberg_term, Heisenberg_term,
                  J_t*Heisenberg_term,J_t*Heisenberg_term,J_t*Heisenberg_term]

tens_arr_opt, lamb_arr_opt = gPEPS.simple_update_optimization(tens_arr_star, lamb_arr_star, ham_arr_star, str_mat_star, 2)

exp_val = 0
for i in 1:9
    global exp_val += gPEPS.expectation_value_two_body(tens_arr_opt, lamb_arr_opt, str_mat_star, ham_arr_star[i], i)
end
exp_val_per_site = exp_val / 6

@info "the energy per site on the star lattice of the above Hamiltonian is e = $(exp_val_per_site)"