using gPEPS
using TensorKit

#=
Here we take the example of the Heisenberg model in 3d from the gPEPS paper (https://arxiv.org/abs/1808.00680).
This example can be used to reproduce the energies of Fig. 5 of the gPEPS paper.
=# 

n = 2
space = ℝ
A_3d = rand(Float64, space^n ⊗ space^n ⊗ space^n, space^n ⊗ space^n ⊗ space^n ⊗ (space^2)')
B_3d = rand(Float64, space^n ⊗ space^n ⊗ space^n, space^n ⊗ space^n ⊗ space^n ⊗ (space^2)')

tens_arr_3d = [A_3d, B_3d]

str_mat_3d = [-2 5 -3 6 -1 4; 5 -2 6 -3 4 -1]

lamb = isomorphism(space^n,space^n)
lamb_arr_3d = [lamb for i in 1:6]

S_x, S_y, S_z, Heisenberg_term, Id_term = gPEPS.create_spin_operators_SU(space_type = space)

ham_arr_3d = [Heisenberg_term for i in 1:6]

tens_arr_opt, lamb_arr_opt = gPEPS.simple_update_optimization(tens_arr_3d, lamb_arr_3d, ham_arr_3d, str_mat_3d, n)

exp_val = 0
for i in 1:6
    global exp_val += gPEPS.expectation_value_two_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, ham_arr_3d[i], i)
end
exp_val_per_site = exp_val / 2
@info "The expectation value per site of the 3d Heisenberg model at bond-dimension d = $n is e = $(exp_val_per_site)."  

m_x = 0
m_x += gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_x, 1)
m_x -= gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_x, 2)

m_y = 0
m_y += gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_y, 1)
m_y -= gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_y, 2)

m_z = 0
m_z += gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_z, 1)
m_z -= gPEPS.expectation_value_one_body(tens_arr_opt, lamb_arr_opt, str_mat_3d, S_z, 2)

m = sqrt((m_x/2)^2 + (m_y/2)^2 + (m_z/2)^2)

@info "The staggered magnetization of the 3d Heisenberg model at bond-dimension d = $n is m = $(m)."