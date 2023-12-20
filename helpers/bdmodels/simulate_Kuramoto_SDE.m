function [out_struct] = simulate_Kuramoto_SDE(in_struct)
    addpath(genpath('/home/govindas/bdtoolbox-2022b'));
    addpath(genpath('/home/govindas/network_estimation'));

    % model
    n = size(in_struct.Kij, 1);
    sys = Kuramoto_SDE(in_struct.Kij);
    
    % parameter values
    if isfield(in_struct, 'beta')
        sys.pardef = bdSetValue(sys.pardef, 'beta', in_struct.beta);
    end
    if isfield(in_struct, 'k')
        sys.pardef = bdSetValue(sys.pardef, 'k', in_struct.k);
    end
    if isfield(in_struct, 'A')
        sys.pardef = bdSetValue(sys.pardef, 'A', in_struct.A);
    end
    if isfield(in_struct, 'B')
        sys.pardef = bdSetValue(sys.pardef, 'B', in_struct.B);
    end
    if isfield(in_struct, 'C')
        sys.pardef = bdSetValue(sys.pardef, 'C', in_struct.C);
    end
    if isfield(in_struct, 'd')
        sys.pardef = bdSetValue(sys.pardef, 'd', in_struct.d);
    end
    if isfield(in_struct, 'omega')
        sys.pardef = bdSetValue(sys.pardef, 'omega', in_struct.omega);
    end
    if isfield(in_struct, 'Iamp')
        sys.pardef = bdSetValue(sys.pardef, 'Iamp', in_struct.Iamp);
    end
    if isfield(in_struct, 'tau')
        sys.pardef = bdSetValue(sys.pardef, 'tau', in_struct.tau);
    end
    if isfield(in_struct, 'T')
        sys.pardef = bdSetValue(sys.pardef, 'T', in_struct.T);
    end
    if isfield(in_struct, 's')
        sys.pardef = bdSetValue(sys.pardef, 's', in_struct.s);
    end
    if isfield(in_struct, 'sigma')
        sys.pardef = bdSetValue(sys.pardef, 'sigma', in_struct.sigma);
    end
    
    % time domain
    sys.tspan = in_struct.tspan;
    
    % solver options
    sys.sdeoption.RelTol = 1e-6;
    sys.sdeoption.InitialStep = 0.001;
    sys.sdeoption.MaxStep = 0.005;
    sys.sdeoption.NoiseSources = 2*n;
    if isfield(in_struct, 'randn')
        sys.sdeoption.randn = in_struct.randn;
    end
    
    % initial conditions
    % sys.vardef(1).value = 0.0;
    
    % solver
    sol = bdSolve(sys, sys.tspan, sys.sdesolver{1});
    % [sys, sol] = bdEvolve(sys, 5, sys.tspan, @ode23);
    
    % extract the results
    t = in_struct.teval;
    Y = bdEval(sol, t);
    y = ts(Y,in_struct.Kij,in_struct.beta);
    
    out_struct.t = t;
    out_struct.x = y;
   
end

function [y] = ts(Y,Kij,beta)
    n = size(Kij, 1);
    x = Y(1:n, :);
    theta = Y(n+1:end, :);
    y = x + beta .* sin(theta);
end