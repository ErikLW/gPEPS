function expectation_value_two_body(tens_arr, λ, S, gate, i)

    column = S[:,i]

    #find the indices in each column of the Structure matrix that are nonzero. These label the PEPS tensors relevant for this step:
    ind_relevant_A1 = findall(x -> ispositive(x), column)[1]
    ind_relevant_A2 = findall(x -> isnegative(x), column)[1] #note here we always choose one positive and one negative number in each column. This tells us the outgoing and ingoing index.

    A1 = tens_arr[ind_relevant_A1] 
    A2 = tens_arr[ind_relevant_A2] #are these relevant PEPS tensors

    A1_edge_ind = S[ind_relevant_A1, i]
    A2_edge_ind = S[ind_relevant_A2, i] #are the index of the relevant tensors that correspond to the edge that that we are currently treating

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

    _ , N_1 = contract_tensors([adjoint(A1), A1], [[i for i in 1:rank(A1)], vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):rank(A1)])])

    _ , N_2 = contract_tensors([A2, adjoint(A2)], [[i for i in 1:rank(A2)], vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):rank(A2)])])
    #display(N_1)
    #display(N_2)
    #display(λ[i])
    _, norm = contract_tensors([N_1, N_2, adjoint(λ[i]), λ[i]], [[1,2],[3,4],[4,1],[2,3]])

    #display(norm)

    _ , M_1 = contract_tensors([adjoint(A1), A1], [[i for i in 1:rank(A1)], vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):(rank(A1)-1)], [- rank(A1)])])

    _ , M_2 = contract_tensors([A2, adjoint(A2)], [[i for i in 1:rank(A2)], vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):(rank(A2)-1)], [- rank(A2)])])

    _, rho = contract_tensors([M_1, M_2, adjoint(λ[i]), λ[i]],[[1,2,3,4],[5,6,7,8],[7,1],[3,5]])

    #display(rho)
    _, val = contract_tensors([rho, gate],[[1,2,3,4],[1,4,2,3]])

    #display(val)

    exp_val = val / norm
    return exp_val
end

function expectation_value_one_body(tens_arr, λ, S, op, i)

    row = S[i,:]
    A = tens_arr[i]
    #find the indices in each row of the Structure matrix that are nonzero. These label the lambda tensors relevant for this step:
    ind_relevant_λ = findall(x -> !iszero(x), row)[1]

    #A_edge_ind = S[i, ind_relevant_λ] 

    #absorb the λ-matrices into the tensors A
    for jj in 1:length(ind_relevant_λ) #here we are looping to absorb all the lambda matrices
        
        ind = S[i, ind_relevant_λ[jj]] #ind is the index of the tensor A that we want to put the lambda matrix on to. 
        
        _ , Aλ = contract_tensors([A, λ[ind_relevant_λ[jj]]], [[i for i in 1:rank(A)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

        #here we create the relevant permutation tuple
        permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A)], [j for j in abs(ind):(rank(A)-1)])
        permutelike = Tuple(permutelike)  
        
        A = permute(Aλ, permutelike);
    end

    _, rho = contract_tensors([A, adjoint(A)],[ vcat([j for j in 1:(rank(A)-1)], [rank(A)]),vcat([j for j in 1:(rank(A)-1)], [-rank(A)])])
    _, norm = contract_tensors([A, adjoint(A)],[[j for j in 1:rank(A)],[j for j in 1:rank(A)]])

    _, val = contract_tensors([rho, op],[[1,2],[2,1]])
    exp_val = val/norm
    return exp_val
end
