% Reduced Wong-Wang model of neural activity.
%
% The model shunts the external input with state variable to give an upper
% bound to values the variable takes.
function sys = R_Wong_Wang_SDE(Kij)    
    % number of nodes
    n = size(Kij, 1);
    
    % Handle to SDE function
    sys.sdeF = @sdeF;
    sys.sdeG = @sdeG;
    
    % SDE parameters
    sys.pardef = [
        struct('name','Kij',   'value',Kij, 'lim',[-10, 10])
        struct('name','a', 'value',0.270, 'lim',[0, 1])
        struct('name','b', 'value',0.108, 'lim',[0, 1])
        struct('name','d', 'value',154, 'lim',[0, 5])
        struct('name','gamma', 'value',0.641, 'lim',[0, 1])
        struct('name','tau_s', 'value',100, 'lim',[0, 150])
        struct('name','w', 'value',0.6, 'lim',[0, 1])
        struct('name','J_N', 'value',0.2609, 'lim',[0, 1]) 
        struct('name','Iamp', 'value',1, 'lim',[0, 10])
        struct('name','tau', 'value',10, 'lim',[0, 100])
        struct('name','T', 'value',25, 'lim',[0, 100])
        struct('name','s', 'value',zeros(n,1), 'lim',[0, 1])
        struct('name','sigma','value',0,'lim',[0,1])
        ];
               
    % SDE variables        
    sys.vardef = [
        struct('name','S', 'value',zeros(n,1), 'lim',[-1, 1])        
    ];
    
    % time span
    sys.tspan = [0 100];
    sys.tstep = 0.001;

    % sde options
    sys.sdesolver = {@sdeEM, @sdeSH};
    sys.sdeoption.RelTol = 1e-6;
    sys.sdeoption.InitialStep = 0.001;
    sys.sdeoption.MaxStep = 0.005;
    sys.sdeoption.NoiseSources = n;

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
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};

    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';
      
end

function Stimulus(ax,t,sol,Kij,a,b,d,gamma,tau_s,w,J_N,Iamp,tau,T,s,sigma)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,A, ...: model parameters 
    % Reconstruct the stimulus used by odefun
    Iapp = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,iapp] = sdeF( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            Kij,a,b,d,gamma,tau_s,w,J_N,Iamp,tau,T,s,sigma ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end

% deterministic part
function [F, Iapp] = sdeF(t,Y,Kij,a,b,d,gamma,tau_s,w,J_N,Iamp,tau,T,s,sigma)
    % extract incoming variables
    Y = reshape(Y,[],1);
    S = Y(:, 1);
    
    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end

    % equations
    x = net_activity(S,Kij,w,J_N,Iapp);
    dS = -S./tau_s + gamma.*(1 - S).*H(x,a,b,d);
         
    % return
    F = [dS];
end

function x = net_activity(S,Kij,w,J_N,Iapp)
    x = w .* J_N .* S + J_N .* Kij * S + Iapp;
end

function H_x = H(x,a,b,d)
    H_x = (a.*x - b) ./ (1 - exp(-d .* (a.*x - b)));
    % this is a non-negative filter, x is always >= 0. => S cannot be < 0.
end

% stochastic part
function [G] = sdeG(t,Y,Kij,a,b,d,gamma,tau_s,w,J_N,Iamp,tau,T,s,sigma)
    G = sigma .* eye(size(Kij, 1));
end