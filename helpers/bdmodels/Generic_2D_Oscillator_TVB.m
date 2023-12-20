function sys = Generic_2D_Oscillator_TVB()
    % Handle to ODE function
    sys.odefun = @odefun;
    
    % SDE parameters
    sys.pardef = [
        struct('name','tau_s', 'value',1, 'lim',[1, 5])
        struct('name','a', 'value',-2, 'lim',[-5, 5])
        struct('name','b', 'value',-10, 'lim',[-20, 15])
        struct('name','c', 'value',0, 'lim',[-10, 10])
        struct('name','d', 'value',0.2, 'lim',[0, 1])
        struct('name','e', 'value',3, 'lim',[-5, 5])
        struct('name','f', 'value',1, 'lim',[-5, 5])
        struct('name','g', 'value',0, 'lim',[-5, 5])
        struct('name','alpha', 'value',1, 'lim',[-5, 5])
        struct('name','beta', 'value',1, 'lim',[-5, 5])
        struct('name','gamma', 'value',-1, 'lim',[-1, 1])
        struct('name','I', 'value',0, 'lim',[-5, 5])
    ];
    
    % SDE variables 
    sys.vardef = [
        struct('name','v', 'value',0, 'lim',[-5, 5])
        struct('name','w', 'value',0, 'lim',[-15, 15])
    ];
    
    % time span
    sys.tspan = [0 100];
    sys.tstep = 0.001;

    % ode options
    sys.odesolver = {@ode23s,@ode45,@ode23,@odeEul};
    sys.odeoption.RelTol = 1e-6;
    sys.odeoption.InitialStep = 0.001;
    sys.odeoption.MaxStep = 0.005;
    
    % Equations
    sys.panels.bdLatexPanel.title = 'Equations';
    sys.panels.bdLatexPanel.latex = {
        'Linear model equations';
        '';
        '$\frac{d \, x_i}{dt} = -A_i \, x_i + \left( B_i - C_i \, x_i^2 \right) \left[ f \left( \sum_{j = 1}^{n} K_{ij} x_j \right) + s_i \, I(t) \right], \quad \forall \, i = 1, \cdots, n$';
        '';
    };

    % Display panels -- for GUI
    sys.panels.bdTimePortrait = [];
    sys.panels.bdPhasePortrait = [];
    sys.panels.bdSolverPanel = [];
    %sys.panels.bdAuxiliary.auxfun = {@Stimulus};

    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';

end

% deterministic part
function [F] = odefun(t,Y,tau_s,a,b,c,d,e,f,g,alpha,beta,gamma,I)
    Y = reshape(Y, [], 2);
    v = Y(:, 1);
    w = Y(:, 2);

    % equations
    dv = d .* tau_s .* (-f.*v.^3 + e.*v.^2 + g.*v + alpha.*w + gamma.*I);
    dw = (d ./ tau_s) .* (c.*v.^2 + b.*v - beta.*w + a);

    F = [dv; dw];
end