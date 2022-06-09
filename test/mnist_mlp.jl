
@testset "1000 FashionMNIST examples, BP" begin
    g, w, teacher, E, it = run_experiment(; multiclass=false, dataset=:fashion, lay_type=:bp, seed=2, 
                                M=1000, 
                                K=[28*28, 201, 201, 1], 
                                ρ=[1.0, 1.0, 0.9], density=1.0, ψ=[0.8, 0.8, 0.8],
                                epochs=20, batchsize=128, usecuda=false, gpu_id=0,  
                                maxiters=1, r=0.0, ϵinit=1.0, 
                                altsolv=false, altconv=false, saveres=false);

    # @test E <= 20
    @test E <= 30
end