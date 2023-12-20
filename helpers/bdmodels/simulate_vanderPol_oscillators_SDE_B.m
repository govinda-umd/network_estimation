function [out_struct] = simulate_vanderPol_oscillators_SDE_B(in_struct)
    addpath(genpath('/home/govindas/bdtoolbox-2022b'));
    addpath(genpath('/home/govindas/network_estimation'));

    % model
    n = size(in_struct.W, 1);
    sys = vanderPol_oscillators_SDE_B(in_struct.W);
    
    % parameter values
    if isfield(in_struct, 'a')
        sys.pardef = bdSetValue(sys.pardef, 'a', in_struct.a);
    end
    if isfield(in_struct, 'b')
        sys.pardef = bdSetValue(sys.pardef, 'b', in_struct.b);
    end
    if isfield(in_struct, 'c')
        sys.pardef = bdSetValue(sys.pardef, 'c', in_struct.c);
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
    if isfield(in_struct, 'x_init')
        sys.vardef = bdSetValues(sys.vardef, in_struct.x_init);
    end
    
    % solver
    sol = bdSolve(sys, sys.tspan, sys.sdesolver{1});
    % [sys, sol] = bdEvolve(sys, 5, sys.tspan, @ode23);
    
    % extract the results
    t = in_struct.teval;
    x = bdEval(sol, t);

    out_struct.t = t;
    out_struct.x = x(1:n, :);
    out_struct.w = x(n+1:2*n, :);
    
%     % plot
%     plot(t, x);
%     grid("on");

end