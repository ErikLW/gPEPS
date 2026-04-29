function elementwise_inverse_diag_SU(A::TensorMap)
    diagonal = diag(A.data)
    A_elementwise_inv = diagm(1 ./ diagonal)
    
    return TensorMap(A_elementwise_inv, codomain(A) ← domain(A))
end

function elementwise_sqrt_diag_SU(A::TensorMap)
    diagonal = diag(A.data)
    A_elementwise_sqrt = diagm(sqrt.(diagonal))
    
    return TensorMap(A_elementwise_sqrt, codomain(A) ← domain(A))
end

function convert_SU_to_PEPS_network(A, λ, S; lattice = :square)
    #=
    In this function we need to first split the λ-matrices in 2 by taking the sqrt.
    Then we need to go through all tensors in A and attach the sqrt-λ-matrices to the bonds. 
    This will then give us the output in terms of a PEPS network without λ-matrices on the bonds.
    =#
    
    A = convert(Vector{TensorMap}, A)
    
    λ_sqrt = elementwise_sqrt_diag_SU.(λ)
    
    for (i, row) in enumerate(eachrow(S))
        #find the the λ-matrices that attach to the i'th A-tensor
        ind_relevant_λ = findall(x -> !iszero(x), S[i, 1:end])
        
        for jj in 1:length(ind_relevant_λ)
            
            ind = S[i, ind_relevant_λ[jj]]
            
            _ , Aλ = contract_tensors([A[i], λ_sqrt[ind_relevant_λ[jj]]], [[i for i in 1:rank(A[i])],[ind, -ind]]) #this contracts the two tensors but we still need to put the result back in the correct order

            #here we create the relevant permutation tuple
            permutelike = vcat([i for i in 1:(abs(ind)-1)], [rank(A[i])], [j for j in abs(ind):(rank(A[i])-1)])
            permutelike = Tuple(permutelike)  
            
            A_int = permute(Aλ, permutelike);
            
            #now depending on the lattice at hand, we put the indizes into the right domain/codomain!
            if lattice == :square
                A[i] = permute(A_int, (1,2), (3,4,5))
                A[i] = A[i] / norm(A[i])
            else
                @warn "Beware, you have not specified the specific lattice in the function: 'convert_SU_to_PEPS_network'!"
            end
        end
    end

    A = convert(Vector{typeof(A[1])}, A)
    
    return A

end

function increase_bonddimension_with_SU(A, S, d::Int; pert_strength = 0)
    #=
    this function is aimed at increasing the Bond dimension of PEPS that are not in the SU-form. 
    For this we will take in such a set of PEPS tensors and will take the λ-matrices to be identities. --> Does that work?
    For this to work we will need to supply a structure matrix S as well as a desired bond dimension d as an input. 
    =#
    
    #create a 2-body gate of Identity tensor: 
    #first specify the dimension of the local physical Hilbert space
    d_phys = dim(domain(A[1])[3])
    #Then create a identity matrix and the 2-body identity gate
    σ_0 = TensorMap(Matrix(1.0 * I, d_phys, d_phys), ℂ^d_phys ← ℂ^d_phys)
    @tensor Id_term[(i,j);(k,l)] := σ_0[i,k] * σ_0[j, l]
    random_pert = TensorMap(randn, ℂ^d_phys ⊗ ℂ^d_phys ← ℂ^d_phys ⊗ ℂ^d_phys)
    rand_pert_weighted = pert_strength * random_pert
    H_identities = [Id_term + rand_pert_weighted  for i in 1:size(S)[2]]
    
    
    #create an array of λ-matrices that just contain identities of the input bond dimension.
    #first specify the local bond dimension of the input matrices
    d_virt = dim(codomain(A[1])[1])
    λ_id = TensorMap(Matrix(1.0 * I, d_virt, d_virt), ℂ^d_virt ← ℂ^d_virt)
    λ_identities = [λ_id for i in 1:size(S)[2]]
    
    #perfrom a SU step with the 2-body-identity
    A_grown, λ_grown = simple_update_step(A, λ_identities, H_identities, S, d)
    #display(norm(A_grown[1]))
    #now that we have grown the PEPS in bond dimension we need to put the λ-matrices in λ_grown and the A's in A_grown together to put the output into the form of the input
    A_grown_with_SU = convert_SU_to_PEPS_network(A_grown, λ_grown, S);
    
    return A_grown_with_SU
    
end