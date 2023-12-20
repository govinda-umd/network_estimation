function [out_struct] = simulate_vanderPol_oscillators_ODE_A(in_struct)
    addpath(genpath('/home/govindas/bdtoolbox-2022b'));
    addpath(genpath('/home/govindas/network_estimation'));

    % model
    n = size(in_struct.W, 1);
    sys = vanderPol_oscillators_ODE_A(in_struct.W);
    
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
    
    % time domain
    sys.tspan = in_struct.tspan;
    
    % solver options
    sys.odeoption.RelTol = 1e-6;
    
    % initial conditions
    % sys.vardef(1).value = 0.0;
    
    % solver
    sol = bdSolve(sys, sys.tspan, @ode23);
    % [sys, sol] = bdEvolve(sys, 5, sys.tspan, @ode23);
    
    % extract the results
    t = in_struct.teval;
    x = bdEval(sol, t);

    out_struct.t = t;
    out_struct.x = x(1:n, :);
    
%     % plot
%     plot(t, x);
%     grid("on");

end