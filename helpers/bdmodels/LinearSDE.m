% Linear model of neural activity.
%    dx = [F(t, x)] .* dt + [G(t, x)] .* dWt
%    F(t, x) = -gamma.*x + (x - bl).*(bu - x).*(Inet + Iapp)
%    G(t, x) = sigma .* eye(n)
%
% The model shunts the external input with state variable to give an upper
% bound to values the variable takes.
%
% Example:
%   sys = Linear();         % Construct the system struct.
%   gui = bdGUI(sys);       % Open the Brain Dynamics GUI.
%
function sys = LinearSDE(Kij)    
    % number of nodes
    n = size(Kij, 1);
    
    % Handle to SDE function
    sys.sdeF = @sdeF;
    sys.sdeG = @sdeG;
    
    % SDE parameters
    sys.pardef = [
        struct('name','Kij', 'value',Kij, 'lim',[0, 10])
        struct('name','gamma', 'value',0.5*ones(n,1), 'lim',[0, 100])
        struct('name','bu', 'value',1.0, 'lim',[-5, 5])
        struct('name','bl', 'value',-1.0, 'lim',[-5, 5])
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
    sys.sdesolver = {@sdeSH, @sdeEM};
    sys.sdeoption.RelTol = 1e-6;
    sys.sdeoption.InitialStep = 0.001;
    sys.sdeoption.MaxStep = 0.005;
    sys.sdeoption.NoiseSources = n;

    % Equations
    sys.panels.bdLatexPanel.title = 'Equations';
    sys.panels.bdLatexPanel.latex = {
        'Linear model equations';
        '';
        'd$\textbf{x} = \textbf{F}(t, \textbf{x}) \, $d$t + \textbf{G}(t, \textbf{x}) \, $d$\textbf{w}(t)$';
        '';
        '$\textbf{F}(t, \textbf{x}) = \left[- \gamma \textbf{x} + (\textbf{x} - b_l) (b_u - \textbf{x}) \left( \textbf{K} \, f (\textbf{x}) + \textbf{g}(t) \right) \right]$';
        '';
        '$\textbf{g}(t) = I_{amp}(t) \, \textbf{s}$: external stimulus';
        '';
        '$\textbf{G}(t, \textbf{x}) = \sigma \, \textbf{I}_{n \times n}$: $n$ independent Weiner processes all scaled with std. $\sigma$.';
        '';
        '$\textbf{w}(t)$: $n$ independent Weiner processes';
    };

    % Display panels -- for GUI
    sys.panels.bdTimePortrait = [];
    sys.panels.bdPhasePortrait = [];
    sys.panels.bdSolverPanel = [];
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};
      
end

function Stimulus(ax,t,sol,Kij,gamma,bu,bl,Iamp,tau,T,s,sigma)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,gamma, ...: model parameters 
    % Reconstruct the stimulus used by odefun
    Iapp = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,iapp] = sdeF( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            Kij,gamma,bu,bl,Iamp,tau,T,s,sigma ...
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
function [F, Iapp] = sdeF(t,Y,Kij,gamma,bu,bl,Iamp,tau,T,s,sigma)
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
    Inet = Kij * f(x);

    % system of equations
    dx = -gamma.*x + (x - bl).*(bu - x).*(Inet + Iapp);    

    % return
    F = [dx];
end

% stochastic part
function [G] = sdeG(t,Y,Kij,gamma,bu,bl,Iamp,tau,T,s,sigma)
    G = sigma .* eye(size(Kij,1));
end

% Sigmoid function
function y=f(x)
    y = 1./(1+exp(-x)) - 0.5;
end