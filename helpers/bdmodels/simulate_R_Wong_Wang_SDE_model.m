function [out_struct] = simulate_R_Wong_Wang_SDE_model(in_struct)
    addpath(genpath('/home/govindas/bdtoolbox-2022b'));
    addpath(genpath('/home/govindas/network_estimation'));

    % model
    n = size(in_struct.Kij, 1);
    sys = R_Wong_Wang_SDE(in_struct.Kij);
    
    % parameter values
    if isfield(in_struct, 'a')
        sys.pardef = bdSetValue(sys.pardef, 'a', in_struct.a);
    end
    if isfield(in_struct, 'b')
        sys.pardef = bdSetValue(sys.pardef, 'b', in_struct.b);
    end
    if isfield(in_struct, 'd')
        sys.pardef = bdSetValue(sys.pardef, 'd', in_struct.d);
    end
    if isfield(in_struct, 'gamma')
        sys.pardef = bdSetValue(sys.pardef, 'gamma', in_struct.gamma);
    end
    if isfield(in_struct, 'tau_s')
        sys.pardef = bdSetValue(sys.pardef, 'tau_s', in_struct.tau_s);
    end
    if isfield(in_struct, 'w')
        sys.pardef = bdSetValue(sys.pardef, 'w', in_struct.w);
    end
    if isfield(in_struct, 'J_N')
        sys.pardef = bdSetValue(sys.pardef, 'J_N', in_struct.J_N);
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
    sys.sdeoption.NoiseSources = n;
    if isfield(in_struct, 'randn')
        sys.sdeoption.randn = in_struct.randn;
    end
    
    
    % initial conditions
    % sys.vardef(1).value = 0.0;
    
    % solver
    sol = bdSolve(sys, sys.tspan, sys.sdesolver{1});
    
    % extract the results
    t = in_struct.teval;
    x = bdEval(sol, t);

    out_struct.t = t;
    out_struct.x = x;
    
end