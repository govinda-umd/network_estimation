function [out_struct] = simulate_linear_model(in_struct)
    addpath(genpath('/home/govindas/bdtoolbox-2022b'));
    addpath(genpath('/home/govindas/network_estimation'));

    % model
    sys = Linear(in_struct.Kij);
    
    % parameter values
    if isfield(in_struct, 'gamma')
        sys.pardef = bdSetValue(sys.pardef, 'gamma', in_struct.gamma);
    end
    if isfield(in_struct, 'bu')
        sys.pardef = bdSetValue(sys.pardef, 'bu', in_struct.bu);
    end
    if isfield(in_struct, 'bl')
        sys.pardef = bdSetValue(sys.pardef, 'bl', in_struct.bl);
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
    
    % time domain
    sys.tspan = in_struct.tspan;
    
    % solver options
    sys.odeoption.RelTol = 1e-6;
    
    % initial conditions
    % sys.vardef(1).value = 0.0;
    
    % solver
    sol = bdSolve(sys, sys.tspan, @ode23s);
    % [sys, sol] = bdEvolve(sys, 5, sys.tspan, @ode23);
    
    % extract the results
    t = in_struct.teval;
    x = bdEval(sol, t);

    out_struct.t = t;
    out_struct.x = x;
    
%     % plot
%     plot(t, x);
%     grid("on");

end