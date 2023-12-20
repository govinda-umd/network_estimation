% Linear model of neural activity.
%    dx = F(t, x) .* dt + G(t, x) .* dN(t)
%    F(t, x) = -A.*x + (x - -(B)).*(B - x).*(Inet + Iapp)
%    G(t, x) = sigma .* eye(n)
%
% The model shunts the external input with state variable to give an upper
% bound to values the variable takes.
%
% Example:
%   sys = Linear();         % Construct the system struct.
%   gui = bdGUI(sys);       % Open the Brain Dynamics GUI.
%
function sys = Nonlinear_1D_ODE(W)    
    % number of nodes
    n = size(W, 1);
    
    % Handle to SDE function
    sys.odefun = @odefun;
    
    % SDE parameters
    sys.pardef = [
        struct('name','W', 'value',W, 'lim',[-10, 10])
        struct('name','A', 'value',1.0*ones(n,1), 'lim',[0, 10])
        struct('name','B', 'value',0.75, 'lim',[0, 5])
        struct('name','C', 'value',1.0, 'lim',[0, 5])
        struct('name','Iamp', 'value',1, 'lim',[0, 10])
        struct('name','tau', 'value',10, 'lim',[0, 10])
        struct('name','T', 'value',25, 'lim',[0, 10])
        struct('name','s', 'value',zeros(n,1), 'lim',[0, 1])
        ];
               
    % SDE variables        
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[-1, 1])        
    ];
    
    % time span
    sys.tspan = [0 100];
    sys.tstep = 0.001;

    % sde options
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
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};

    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';
      
end

function Stimulus(ax,t,sol,W,A,B,C,Iamp,tau,T,s)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,A, ...: model parameters 
    % Reconstruct the stimulus used by odefun
    Iapp = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,iapp] = odefun( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            W,A,B,C,Iamp,tau,T,s ...
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
function [F, Iapp] = odefun(t,Y,W,A,B,C,Iamp,tau,T,s)
    % extract incoming variables
    Y = reshape(Y,[],1);
    x = Y(:, 1);
    
    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end

    % neighbor nodes' influences
    Inet = f(W * x);

    % system of equations
    dx = -A.*x ...
        +(B - C.*x.^2).*(Inet + Iapp); 
         
    % return
    F = [dx];
end

% Sigmoid function
function y=f(x)
    y = 1./(1+exp(-x)) - 0.5;
    %y = tanh(x);
end