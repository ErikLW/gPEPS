function rank(A::TensorMap)
    rank_dom = length(dims(codomain(A)))
    rank_codom = length(dims(domain(A)))
    
    return rank_dom + rank_codom
end

function leftorth(A::TensorMap, i1::Tuple, i2::Tuple)
    #display("hello")
    A_perm = TensorKit.permute(A,(i1,i2))
    
    Q,R = TensorKit.left_orth(A_perm)
    return Q,R
end

function rightorth(A::TensorMap, i1::Tuple, i2::Tuple)
    #display("hello2")
    A_perm = TensorKit.permute(A,(i1,i2))
    
    L,Q = TensorKit.right_orth(A_perm)
    return L,Q
end

truncdim(d::Int) = truncrank(d::Int)

function tsvd(A::TensorMap, i1::Tuple, i2::Tuple; trunc::Any = nothing)

    A_perm = TensorKit.permute(A,(i1,i2))

    U, S, Vd = svd_trunc(A_perm; trunc)

    return U, S, Vd
end

function eigh(A::TensorMap, i1::Tuple, i2::Tuple)
    A_perm = TensorKit.permute(A,(i1,i2))

    D, V = eigh_full(A_perm)

    return D, V
end


#function contract_tensors(tensor_list::Vector{<:AbstractTensorMap}, index_list::Vector{Vector{K}}) where {K}
function contract_tensors(tensor_list, index_list::Vector{Vector{K}}) where {K}

    #= This function takes in an array of tensors (tensor_list) and and array of index lists (index_list). 
    Whenever two indizes in the index list are identical they are summed over. The remaining, unique indices that have not been summed over,
    are ordered from first to last in the index list. =#


    #= why is this slower than the version of Wladi? Would expect to be similar...
    fl_index_list = Iterators.flatten(index_list)
    n_count = StatsBase.countmap(fl_index_list)
    defects = findall(x -> x>2, n_count)
    isempty(defects) || error("Indices $(defects) occure more than twice.")
    double_occurence = findall(x -> x == 2, n_count)
    single_occurence = findall(x -> x == 1, n_count)
    =#
    unique_indices = K[]
    double_indices = K[]
    flatIndexList = collect(Iterators.flatten(index_list))#vcat(indexList...)

    while !(isempty(flatIndexList))
        el = popfirst!(flatIndexList)
        if !(el in flatIndexList) && !(el in double_indices)
            append!(unique_indices, el)
        else
            append!(double_indices, el)
        end
    end

    contract_list = map(index_list) do list
        map(list) do pp
            #return pp in unique_indices ? -findall(isequal(pp), unique_indices)[1] : findall(isequal(pp), double_indices)[1]
            return pp in unique_indices ? -findfirst(isequal(pp), unique_indices) : findfirst(isequal(pp), double_indices)
        end
    end
    return unique_indices, @ncon(tensor_list, contract_list)
end


ispositive = x -> x > 0
isnegative = x -> x < 0



function simple_update_step(A, λ_in, H, S, d; gauge_fixing = false)
    #=
    testing please
    The simple update step works by looping over the individual edges in the unit cell and updating them as in fig. 2 in the gPEPS paper.
    For every edge in the unit cell one must specify a few things that we need to know for the 
    simple update step on this specific edge:
    1. The PEPS tensors on the vertices of that edge - We further need to know which virtual index this step is updating
    2. The lambda tensor of the edge that is being updated
    3. The remaining lambda tensors that are associated with the PEPS tensors that are being updated but do not live on the edge which is updated
    4. The Hamiltonian-Term that is needed for this specific edge.
    =#
    
    #=
    in the last step of the simple update we need to multiply with the inverse of the λ-matrices. 
    Thus we create an array analogous to λ that contains the inverse matrices.
    These inverse matrices are then updated simultaneously with the λ-matrices during the algorithm
    =#
    
    A = convert(Vector{TensorMap}, A) #this is just a change of type to avoid type-conflicts. Don't worry about this.
    
    λ = copy(λ_in) #we don't want to everwrite the input arguments, so we create a copy called λ that we overwrite step by step and then return as a result.

    #during the algorithm we need the inverse of the diagonal matrices. Here we create this array.
    λ_inv = []
    for k in eachindex(λ)
        #push!(λ_inv, elementwise_inverse_diag_SU(λ[k]))
        push!(λ_inv, inv(λ[k]))
    end
    
    #Here I tested a version of fixing a gauge (approximately). Ignore this.
    if gauge_fixing == true    
        perform_gauge_fixing(A, λ, S)
    end
    
    #=Here we start the loop over the edges of the lattice. The lattice is specified by the structure matrix S. The columns of the structure matrix
    correspond to these edges.=#
    for (i, column) in enumerate(eachcol(S)) #i is the runnung variable of the loop (starting at 1 and going up to the number of columns in S). column is the column of S corresponding to the current edge.
        
        #find the indices in each column of the Structure matrix that are nonzero. These label the PEPS tensors relevant for this step:
        ind_relevant_A1 = findall(x -> ispositive(x), column)[1]
        ind_relevant_A2 = findall(x -> isnegative(x), column)[1] #note here we always choose one positive and one negative number in each column. This tells us the outgoing and ingoing index.
        
        A1 = A[ind_relevant_A1] 
        A2 = A[ind_relevant_A2] #are these relevant PEPS tensors
        
        A1_edge_ind = S[ind_relevant_A1, i]
        A2_edge_ind = S[ind_relevant_A2, i] #are the index of the relevant tensors that correspond to the edge that that we are currently treating
        
        #=now we go through the rows that correspond to our two relevant tensors (tensors of the edge of the lattice that we are updating currently)
        and we find the other lambda-tensors that are connected to our relevant tensors but are not on the edge that we are currently treating=#
        
        #these give the indices for all the λ-matrices, which are relevant for the two tensors that live on the vertices of the edge in question
        ind_relevant_λ1 = findall(x -> !iszero(x), S[ind_relevant_A1, 1:end])
        deleteat!(ind_relevant_λ1, findall(x->x==i, ind_relevant_λ1)) #here we delete from this list again the index of the corresponding to the edge that we are currently treating.
        
        ind_relevant_λ2 = findall(x -> !iszero(x), S[ind_relevant_A2, 1:end])
        deleteat!(ind_relevant_λ2, findall(x->x==i, ind_relevant_λ2))#here we delete from this list again the index of the corresponding to the edge that we are currently treating.

        #the legs which the λ-matrices, that are specified by the above indizes, are connected to in the tensor in question are e.g. S[ind_relevant_A[1], ind_relevant_λ1[1 or any index]]
                
        #STEP (i) of fig. 2 of the gPEPS paper:

        #the first action is to absorb the λ-matrices, that are not on the current edge into the tensors A1
        for jj in 1:length(ind_relevant_λ1) #here we are looping to absorb all the lambda matrices
            
            ind = S[ind_relevant_A1, ind_relevant_λ1[jj]] #ind is the index of the tensor A1 that we want to put the lambda matrix on to. 
            
            _ , A1λ = contract_tensors([A1, λ[ind_relevant_λ1[jj]]], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
            permutelike = Tuple(permutelike)  
            
            A1 = permute(A1λ, permutelike);
        end
        
        
        #here we repeat the same as above for the second tensor (A2) of the edge that we are treating.
        for jj in 1:length(ind_relevant_λ2)
            
            ind = S[ind_relevant_A2, ind_relevant_λ2[jj]]
            
            _ , A2λ = contract_tensors([A2, λ[ind_relevant_λ2[jj]]], [[i for i in 1:rank(A2)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
            permutelike = Tuple(permutelike)  
            
            A2 = permute(A2λ, permutelike);
            
        end
        
        #STEP (ii) of fig. 2 of the gPEPS paper: we don't need to do this.

        #STEP (iii) of fig. 2 of the gPEPS paper:

        #now we perform a QR / LQ decomposition on A1 and A2 
        #start with A1: we first need to specify which indizes should be on the left and the right side
        right_ind_P1 = (rank(A1), abs(A1_edge_ind)) 
        left_ind_P1 = [i for i in 1:rank(A1)]
        deleteat!(left_ind_P1, (abs(A1_edge_ind), rank(A1)))
        left_ind_P1 = Tuple(left_ind_P1)
        
        #with this index partition we perform the QR decomp
        Q1, R = leftorth(A1, left_ind_P1, right_ind_P1)
        
        #now do the same for P2
        left_ind_P2 = (abs(A2_edge_ind), rank(A2))
        right_ind_P2 = [i for i in 1:rank(A2)]
        deleteat!(right_ind_P2, (abs(A2_edge_ind), rank(A2)))
        right_ind_P2 = Tuple(right_ind_P2)
               
        #perform and LQ decomp for P2
        L, Q2 = rightorth(A2, left_ind_P2, right_ind_P2)

        #STEP (iv & v) of fig. 2 of the gPEPS paper:

        #now we want to perform the absorption of the Hamiltonian term
        #THIS SHOULD BE DONE WITH THE @TENSOR MACRO! FASTER!
        _ , Θ = contract_tensors([R, H[i], L, λ[i]], [[1,2,3],[7,8,2,5],[4,5,6],[3,4]]) #this contracts R, L, λ[i], and the H[i] 
        
        #STEP (vi) of fig. 2 of the gPEPS paper: we don't need to group legs

        #STEP (vii & viii) of fig. 2 of the gPEPS paper: 
        #now perform an SVD on the Θ-tensor yielding the new R- , λ- and L-tensors (Note here that we already truncate to the bond dimension d!)
        R_new, λ_new, L_new = tsvd(Θ, (1,2), (3,4), trunc = truncdim(d))
        
        #STEP (ix) of fig. 2 of the gPEPS paper: 
        #we can now put the tensors Q1 and Q2 together with the new R- and L- tensors.
        #Here we start by doing this for P1
        _, P1_new = contract_tensors([Q1, R_new], [[i for i in 1:(rank(A1)-1)], [j for j in (rank(A1)-1):(rank(A2)+1)]])
        
        #However we must revert them back to the correct ordering.
        permutelike_1 = [i for i in 1:(abs(A1_edge_ind) -1)]
        permutelike_2 = [rank(A1)]
        permutelike_3 = [j for j in abs(A1_edge_ind):rank(A1)-1]
        permutelike_complete = vcat(permutelike_1, permutelike_2, permutelike_3)
        permutelike_complete = Tuple(permutelike_complete)  
            
        A1 = permute(P1_new, permutelike_complete);
        
        #now we do it for P2
        _, P2_new = contract_tensors([L_new, Q2], [[i for i in 1:3], [j for j in 3:(rank(A2)+1)]])
        
        #analogously to the above case of P1 we now put the legs in the correct position again
        permutelike_1 = [i for i in 3:(3+abs(A2_edge_ind)-2)]
        permutelike_2 = [1]
        permutelike_3 = [j for j in (3+abs(A2_edge_ind)-1:rank(A2))]
        permutelike_4 = [2]
        permutelike_complete = vcat(permutelike_1, permutelike_2, permutelike_3, permutelike_4)
        permutelike_complete = Tuple(permutelike_complete)  

        #P2_new_reordered = permute(P2_new, permutelike_complete)
        A2 = permute(P2_new, permutelike_complete)

        #STEP (x) of fig. 2 of the gPEPS paper: 
        #now all that is left is to remove the λ-matrices that we multiplied on the tensors in the first step by multiplying with the corresponding inverses
        #This works completely analogous to the multiplication with the λ-matrices in the first step of the algorithm.
        
        for jj in 1:length(ind_relevant_λ1)
            
            ind = S[ind_relevant_A1, ind_relevant_λ1[jj]]
            
            _ , P1λ_inv = contract_tensors([A1, λ_inv[ind_relevant_λ1[jj]]], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
            permutelike = Tuple(permutelike)  
            
            A1 = permute(P1λ_inv, permutelike);
        end
        
        for jj in 1:length(ind_relevant_λ2)
            
            ind = S[ind_relevant_A2, ind_relevant_λ2[jj]]
            
            _ , P2λ_inv = contract_tensors([A2, λ_inv[ind_relevant_λ2[jj]]], [[i for i in 1:rank(A2)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order
        
            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
            permutelike = Tuple(permutelike)  

            A2 = permute(P2λ_inv, permutelike);


            
            
        end

        #now all steps are performed and we move on to update the relevant variables!
        A[ind_relevant_A1] = A1 / norm(A1) 
        A[ind_relevant_A2] = A2 / norm(A2)
        
        λ[i] = λ_new / norm(λ_new)
        #λ_inv[i] = elementwise_inverse_diag_SU(λ_new)
        λ_inv[i] = inv(λ_new)
        
        #display("hurra the first loop works")
        
    end

    A = convert(Vector{typeof(A[1])}, A)
    
    return A, λ
    
end

function simple_update_optimization(A_in, λ_in, H, S, d; gauge_fixing = false)
    
    #=
    This functions performs the simple update optimization, meaning it uses imaginary time evolution
    (with the simple update approximation) to find the ground state tensors of a specified Hamiltonian.
    It therefore outputs the array of tensors A and the array of tensors λ that were the result of this 
    simple update procedure.

    INPUT ARGUMENTS:
    A_in:   This is an array of Tensors (TensorMap) that are the initial guess for the Tensors of the PEPS TN.
            Obviously since the A-Tensors sit on the vertices of the lattice one needs to specifiy an A Tensor for every
            unique Tensor in the unit cell.

    λ_in:   This is an array of diagonal Matrices (also of type TensorMap) that are the inital guess. Usually they are choosen to be the identity initially.
            These sit on the edges of the lattice. For every unique edge in the unit cell we have to input one of these. 

    H:      This is an array of two body gates of the Hamiltonian. This has to be specified beforehand.
            These sit on the edges of the lattice. For every unique edge in the unit cell we have to input one of these. 

    S:      This is the Structure matrix as it is defined in the gPEPS paper by Orus.

    d:      This is just an integer that specifies the bond dimension of the PEPS.
    =#

    A = copy(A_in)
    λ = copy(λ_in)
    
    #here in this array we list the decreasing imaginary time-steps we take during the simple update
    δτ_stepsize_array = [1e-1/(10^i) for i in 0:4]
    
    #first specify the dimension of the local physical Hilbert space
    d_phys = dim(space(A[1])[gPEPS.rank(A[1])])
    #d_phys = dim(domain(A[1])[3])

    #Then create a identity matrix and the 2-body identity gate
    if spacetype(A_in[1]) == ComplexSpace
        σ_0 = TensorMap(Matrix(1.0 * I, d_phys, d_phys), ℂ^d_phys ← ℂ^d_phys)
    else
        σ_0 = TensorMap(Matrix(1.0 * I, d_phys, d_phys), ℝ^d_phys ← ℝ^d_phys)
    end

    @tensor Id_term[(i,j);(k,l)] := σ_0[i,k] * σ_0[j, l]
    
    for δτ in δτ_stepsize_array
        #δτ = δτ_stepsize_array[i]
        display("this is the step in the optimization with δτ = $δτ")

        #now adjust the array of Hamiltonian terms by creating a new one Id - δτ*H --> This is what is needed for im. time evolution.
        Id_minus_δτ_H = []
        for i in 1:length(H)
            push!(Id_minus_δτ_H, - δτ * H[i] + Id_term) 
        end
        
        for j in 1:1000
            
            #=In this subfunction we perform one "imaginary time step". This is really just what is described after equation (4) in the 
            gPEPS paper by Orus and is illustrated in the fig. 2 in the same paper.=#
            A, λ_new = simple_update_step(A, λ, Id_minus_δτ_H, S, d; gauge_fixing = gauge_fixing)
            
            #now in order to test the convergence, we compare the new λ-matrices with the old ones. 
            if j > 3 #this is useful, because we are somtimes increasing the bond dimension in which case the cannot compare λ and λ_new
                norm_diff_λ_mat = norm(λ_new - λ)
                if norm_diff_λ_mat < 1e-6
                    display(norm_diff_λ_mat)
                    display("this converged after $j iterations with this imaginary timestep.")
                    #display(norm(A[1]))
                    break
                end
            end
            #after comparison we update the λ-matrices
            λ = λ_new
        end
    end
    
    
    
    #=
    one could put the sqrt of the edgetensors in λ already into the PEPS tensors in A such that the output are the 
    optimal PEPS tensors under the SU
    =#
    return A, λ
end

function create_spin_operators_SU(; space_type = ℝ)
    Dim_loc = 2
    space_loc = space_type^Dim_loc
    
    S_p_mat =  [0 1 ; 0 0]
    S_m_mat =  [0 0 ; 1 0]
    S_z_mat = 0.5 * [1 0 ; 0 -1]
    S_y_mat = 0.5 * [0 -1im ; 1im 0]
    S_x_mat = 0.5 * [0 1 ; 1 0]
    σ_0_mat = [1 0 ; 0 1]
    
    S_p = TensorMap(S_p_mat, space_loc ← space_loc)
    S_m = TensorMap(S_m_mat, space_loc ← space_loc)
    S_z = TensorMap(S_z_mat, space_loc ← space_loc)
    S_y = TensorMap(S_y_mat, space_loc ← space_loc)
    S_x = TensorMap(S_x_mat, space_loc ← space_loc)
    σ_0 = TensorMap(σ_0_mat, space_loc ← space_loc)

    
    #@tensor Ham_term[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + σ_z[i,k] * σ_z[j,l]
    @tensor Ham_term[(i, j);(k, l)] :=  (S_p[i,k] * S_m[j, l] + S_m[i,k] * S_p[j,l])/2 + (S_z[i,k] * S_z[j,l])
    #@tensor Ham_term[(i, j);(k, l)] :=  σ_x[i,k] * σ_x[j, l] - σ_y[i,k] * σ_y[j,l] - σ_z[i,k] * σ_z[j,l]
    @tensor Id_term[(i,j);(k,l)] := σ_0[i,k] * σ_0[j, l]
    
    
    Ham_term = real(Ham_term)
    return S_x, S_y, S_z, Ham_term, Id_term
end