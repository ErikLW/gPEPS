function perform_gauge_fixing(A, λ, S)
    
    #SPLIT THIS INTO SEPERATE FUNCTION!!
    
    #Gauge fixing essentially amounts to creating the x, x_inv, y and y_inv matrices for every bond.
    #thus we have to perfrom the following steps for all edges:
    for (i, column) in enumerate(eachcol(S))

        #display(i)
        #find the indizes in each column of the Structure matrix that are nonzero. These label the PEPS tensors relevant for this step:
        ind_relevant_A1 = findall(x -> ispositive(x), column)[1]
        ind_relevant_A2 = findall(x -> isnegative(x), column)[1]

        A1 = A[ind_relevant_A1] 
        A2 = A[ind_relevant_A2] #are these relevant PEPS tensors

        A1_edge_ind = S[ind_relevant_A1, i]
        A2_edge_ind = S[ind_relevant_A2, i] #are the index of the relevant tensors that correspond to the edge that that we are currently treating

        #=now we go through the rows that correspond to our two relevant tensors and we find the other lambda-tensors that are connected to our relevant PEPS but are
        not on the edge that we are currently treating=#

        #these give the indices for all the λ-matrices, which are relevant for the two tensors that live on the vertices of the edge in question
        ind_relevant_λ1 = findall(x -> !iszero(x), S[ind_relevant_A1, 1:end])
        deleteat!(ind_relevant_λ1, findall(x->x==i, ind_relevant_λ1))

        ind_relevant_λ2 = findall(x -> !iszero(x), S[ind_relevant_A2, 1:end])
        deleteat!(ind_relevant_λ2, findall(x->x==i, ind_relevant_λ2))
        #the legs which the λ-matrices, that are specified by the above indizes, are connected to in the tensor in question are e.g. S[ind_relevant_A[1], ind_relevant_λ1[1 or any index]]


        #the first action is to absorb the λ-matrices, that are not on the current edge into the tensors A1
        A1_prime = 0
        A2_prime = 0

        for jj in 1:length(ind_relevant_λ1)

            ind = S[ind_relevant_A1, ind_relevant_λ1[jj]]
            #display(ind)
            #display(space(A1))
            #display(space(λ[ind_relevant_λ1[jj]]))
            _ , A1λ = contract_tensors([A1, λ[ind_relevant_λ1[jj]]], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
            permutelike = Tuple(permutelike)  

            A1_prime = permute(A1λ, permutelike);
        end



        for jj in 1:length(ind_relevant_λ2)

            ind = S[ind_relevant_A2, ind_relevant_λ2[jj]]

            _ , A2λ = contract_tensors([A2, λ[ind_relevant_λ2[jj]]], [[i for i in 1:rank(A2)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
            permutelike = Tuple(permutelike)  

            A2_prime = permute(A2λ, permutelike);

        end

        #display(A2_edge_ind)
        #now contract bra and ket layer to generate M1 and M2. 
        
        #display(vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):rank(A1_prime)]))
        #display(vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):rank(A2_prime)]))
        
        _ , M_1 = contract_tensors([adjoint(A1_prime), A1_prime], [[i for i in 1:rank(A1_prime)], vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):rank(A1_prime)])])

        _ , M_2 = contract_tensors([A2_prime, adjoint(A2_prime)], [[i for i in 1:rank(A2_prime)], vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):rank(A2_prime)])])

        #now we perform the eigenvalue decomposition here. We know the matrices M_1 & M_2 are hermitian (THEY ARE CERTAINLY NORMAL -- READ ABOUT THE RELATIONSSHIP!)

        #display(space(M_1))
        #display(space(M_2))
        
        #display(ishermitian(permute(M_1, (1,), (2,))))
        D_1, V_1 = eigh(M_1, (1,), (2,))
        #display("spaces of D1 and V1")
        #display(space(D_1))
        #display(space(V_1))
        
        D_2, V_2 = eigh(M_2, (1,), (2,))
        
        #display("spaces of D2 and V2")
        #display(space(D_2))
        #display(space(V_2))

        #sqrt_D_1 = elementwise_sqrt_diag_SU(D_1)
        sqrt_D_1 = sqrt(D_1)
        #sqrt_D_2 = elementwise_sqrt_diag_SU(D_2)
        sqrt_D_2 = sqrt(D_2)

        #now we attach the resulting matrices to the central λ-matrix.
        
        #display(space(sqrt_D_1))
        #display(space(V_1'))
        #display(space(λ[i]))
        #display(space(V_2'))
        #display(space(sqrt_D_2))
        
        λ_prime = sqrt_D_1 * V_1' * λ[i] * V_2 * sqrt_D_2
        #display(space(λ_prime))
        #now SVD the λ_prime-matrix
        #W_1, λ_tilde, W_2_dagger = tsvd(λ_prime)
        W_1, λ_tilde, W_2_dagger = svd_full(λ_prime)

        
        #display(W_1 * λ_tilde * W_2_dagger ≈ λ_prime)
        
        #sqrt_D_1_inv = elementwise_inverse_diag_SU(sqrt_D_1)
        #sqrt_D_2_inv = elementwise_inverse_diag_SU(sqrt_D_2)
        sqrt_D_1_inv = inv(sqrt_D_1)
        sqrt_D_2_inv = inv(sqrt_D_2)
        
        #x = W_1' * sqrt_D_1 * V_1'
        x_inv = V_1 * sqrt_D_1_inv * W_1
        #display(space(x_inv))
        #y = V_2 * sqrt_D_2 * W_2_dagger'
        y_inv = W_2_dagger * sqrt_D_2_inv * V_2'
        
        #Since we already know λ_tilde, we don't need calculate x and y but only x_inv and y_inv. We then put them on A1 and A2 respectively.
        #first for A1
        #display(A1_edge_ind)
        ind = A1_edge_ind
        _ , A1_canon = contract_tensors([A1, x_inv], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order
        #here we create the relevant permutation tuple
        permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
        permutelike = Tuple(permutelike)  
        A1_tilde = permute(A1_canon, permutelike);
        
        #then for A2
        ind = A2_edge_ind
        _ , A2_canon = contract_tensors([A2, y_inv], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order
        #here we create the relevant permutation tuple
        permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
        permutelike = Tuple(permutelike)  
        A2_tilde = permute(A2_canon, permutelike);

        #display(ind_relevant_A2)
        A[ind_relevant_A1] = A1_tilde
        A[ind_relevant_A2] = A2_tilde
        λ[i] = λ_tilde
        
        #=
        A1_prime_test = 0
        A2_prime_test = 0

        for jj in 1:length(ind_relevant_λ1)

            ind = S[ind_relevant_A1, ind_relevant_λ1[jj]]
            #display(ind)
            #display(space(A1))
            #display(space(λ[ind_relevant_λ1[jj]]))
            _ , A1λ = contract_tensors([A1_tilde, λ[ind_relevant_λ1[jj]]], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
            permutelike = Tuple(permutelike)  

            A1_prime_test = permute(A1λ, permutelike);
        end



        for jj in 1:length(ind_relevant_λ2)

            ind = S[ind_relevant_A2, ind_relevant_λ2[jj]]

            _ , A2λ = contract_tensors([A2_tilde, λ[ind_relevant_λ2[jj]]], [[i for i in 1:rank(A2)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
            permutelike = Tuple(permutelike)  

            A2_prime_test = permute(A2λ, permutelike);

        end

        #display(A2_edge_ind)
        #now contract bra and ket layer to generate M1 and M2. 
        
        _ , M_1_test = contract_tensors([adjoint(A1_prime_test), A1_prime_test], [[i for i in 1:rank(A1_prime)], vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):rank(A1_prime)])])

        _ , M_2_test = contract_tensors([A2_prime_test, adjoint(A2_prime_test)], [[i for i in 1:rank(A2_prime)], vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):rank(A2_prime)])])

        display(M_1_test)
        display(M_2_test)
        =#
        
    end

end

function test_gauge_fixing(A, λ, S)
    
    
    for (i, column) in enumerate(eachcol(S))

        display(i)
        #find the indizes in each column of the Structure matrix that are nonzero. These label the PEPS tensors relevant for this step:
        ind_relevant_A1 = findall(x -> ispositive(x), column)[1]
        ind_relevant_A2 = findall(x -> isnegative(x), column)[1]

        A1 = A[ind_relevant_A1] 
        A2 = A[ind_relevant_A2] #are these relevant PEPS tensors

        A1_edge_ind = S[ind_relevant_A1, i]
        A2_edge_ind = S[ind_relevant_A2, i] #are the index of the relevant tensors that correspond to the edge that that we are currently treating

        #=now we go through the rows that correspond to our two relevant tensors and we find the other lambda-tensors that are connected to our relevant PEPS but are
        not on the edge that we are currently treating=#

        #these give the indices for all the λ-matrices, which are relevant for the two tensors that live on the vertices of the edge in question
        ind_relevant_λ1 = findall(x -> !iszero(x), S[ind_relevant_A1, 1:end])
        deleteat!(ind_relevant_λ1, findall(x->x==i, ind_relevant_λ1))

        ind_relevant_λ2 = findall(x -> !iszero(x), S[ind_relevant_A2, 1:end])
        deleteat!(ind_relevant_λ2, findall(x->x==i, ind_relevant_λ2))
        #the legs which the λ-matrices, that are specified by the above indizes, are connected to in the tensor in question are e.g. S[ind_relevant_A[1], ind_relevant_λ1[1 or any index]]
        
        A1_prime_test = 0
        A2_prime_test = 0

        for jj in 1:length(ind_relevant_λ1)

            ind = S[ind_relevant_A1, ind_relevant_λ1[jj]]
            #display(ind)
            #display(space(A1))
            #display(space(λ[ind_relevant_λ1[jj]]))
            _ , A1λ = contract_tensors([A1, λ[ind_relevant_λ1[jj]]], [[i for i in 1:rank(A1)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A1)], [j for j in abs(ind):(rank(A1)-1)])
            permutelike = Tuple(permutelike)  

            A1_prime_test = permute(A1λ, permutelike);
        end



        for jj in 1:length(ind_relevant_λ2)

            ind = S[ind_relevant_A2, ind_relevant_λ2[jj]]

            _ , A2λ = contract_tensors([A2, λ[ind_relevant_λ2[jj]]], [[i for i in 1:rank(A2)],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A2)], [j for j in abs(ind):(rank(A2)-1)])
            permutelike = Tuple(permutelike)  

            A2_prime_test = permute(A2λ, permutelike);

        end

        #display(A2_edge_ind)
        #now contract bra and ket layer to generate M1 and M2. 
        
        _ , M_1_test = contract_tensors([adjoint(A1_prime_test), A1_prime_test], [[i for i in 1:rank(A1_prime_test)], vcat([i for i in 1:(abs(A1_edge_ind) -1)], [- abs(A1_edge_ind)], [i for i in (abs(A1_edge_ind)+1):rank(A1_prime_test)])])

        _ , M_2_test = contract_tensors([A2_prime_test, adjoint(A2_prime_test)], [[i for i in 1:rank(A2_prime_test)], vcat([i for i in 1:(abs(A2_edge_ind) -1)], [- abs(A2_edge_ind)], [i for i in (abs(A2_edge_ind)+1):rank(A2_prime_test)])])

        display(M_1_test)
        display(M_2_test)
    end

end