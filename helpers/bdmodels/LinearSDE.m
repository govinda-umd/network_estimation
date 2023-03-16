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
function sys = LinearSDE(W)    
    % number of nodes
    n = size(W, 1);
    
    % Handle to SDE function
    sys.sdeF = @sdeF;
    sys.sdeG = @sdeG;
    
    % SDE parameters
    sys.pardef = [
        struct('name','W', 'value',W, 'lim',[0, 10])
        struct('name','A', 'value',0.5*ones(n,1), 'lim',[0, 100])
        struct('name','B', 'value',1.0, 'lim',[-5, 5])
        struct('name','Iamp', 'value',1, 'lim',[0, 10])
        struct('name','tau', 'value',1, 'lim',[0, 10])
        struct('name','T', 'value',2, 'lim',[0, 10])
        struct('name','s', 'value',zeros(n,1), 'lim',[0, 10])
        struct('name','sigma', 'value',0, 'lim',[0, 1])
        ];
               
    % SDE variables        
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[0, 1])        
    ];
    
    % time span
    sys.tspan = [0 20];
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
        '$d x_i = F_i \left( t, \textbf{x} \right) \, dt + G_i \left( t, \textbf{x} \right) \, dN_i (t)$';
        '';
        '$F_i \left( t, \textbf{x} \right) = -A \, x_i + (B - x_i) \, (x_i - (-B)) \, \left[ \sum_{j = 1}^{n} f \left( x_j \right) w_{ij} + I_i (t) \right]$';
        '';
        '$G_i \left( t, \textbf{x} \right) = \sigma$';
        '';
        '$N_i (t)$: noise as Weiner process';
        '';
    };

    % Display panels -- for GUI
    sys.panels.bdTimePortrait = [];
    sys.panels.bdPhasePortrait = [];
    sys.panels.bdSolverPanel = [];
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};
      
end

function Stimulus(ax,t,sol,W,A,B,Iamp,tau,T,s,sigma)
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
            W,A,B,Iamp,tau,T,s,sigma ...
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
function [F, Iapp] = sdeF(t,Y,W,A,B,Iamp,tau,T,s,sigma)
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
    Inet = W * f(x);

    % system of equations
    dx = -A.*x +  (B - x).*(Inet + Iapp);    

    % return
    F = [dx];
end

% stochastic part
function [G] = sdeG(t,Y,W,A,B,Iamp,tau,T,s,sigma)
    G = sigma .* eye(size(W,1));
end

% Sigmoid function
function y=f(x)
    %y = 1./(1+exp(-x)) - 0.5;
    y = tanh(x);
end