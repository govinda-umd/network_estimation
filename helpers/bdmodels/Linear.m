% Linear model of neural activity.
%    dx = -A*x + (B - x)*Iapp
%
% The model shunts the external input with state variable to give an upper
% bound to values the variable takes.
%
% Example:
%   sys = Linear();         % Construct the system struct.
%   gui = bdGUI(sys);       % Open the Brain Dynamics GUI.
%
function sys = Linear(Kij)    
    % number of nodes
    n = size(Kij, 1);
    
    % Handle to our ODE function
    sys.odefun = @odefun;
    
    % Our ODE parameters
    sys.pardef = [
        struct('name','Kij', 'value',Kij, 'lim',[0, 10])
        struct('name','gamma', 'value',0.5*ones(n,1), 'lim',[0, 100])
        struct('name','bu', 'value',1.0, 'lim',[-5, 5])
        struct('name','bl', 'value',-1.0, 'lim',[-5, 5])
        struct('name','Iamp', 'value',1, 'lim',[0, 10])
        struct('name','tau', 'value',1, 'lim',[0, 10])
        struct('name','T', 'value',2, 'lim',[0, 10])
        struct('name','s', 'value',zeros(n,1), 'lim',[0, 10])
        ];
               
    % Our ODE variables        
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[0, 1])        
        ];
    
    % time span
    sys.tspan = [0 20];

    % ode options
    sys.odeoption.RelTol = 1e-6;
    sys.odeoption.InitialStep = 0.001;
    sys.odeoption.MaxStep = 0.005;

    % Equations
    sys.panels.bdLatexPanel.title = 'Equations';
    sys.panels.bdLatexPanel.latex = {
        'Linear model equations';
        '';
        '$x_{i}^{\prime} = - \gamma x_{i} + (x_{i} - b_l) (b_u - x_{i}) \left( \sum_{j=1}^{n} w_{ij} \, f (x_{j}) + g_{i}(t) \right)$';
        '';
        '$g_{i}(t) = s_{i} \, I(t)$: external stimulus';
    };

    % Display panels -- for GUI
    sys.panels.bdTimePortrait = [];
    sys.panels.bdPhasePortrait = [];
    sys.panels.bdSolverPanel = [];
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};
      
end

function Stimulus(ax,t,sol,Kij,gamma,bu,bl,Iamp,tau,T,s)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,gamma, ...: model parameters
    % Reconstruct the stimulus used by odefun
    Iapp = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,iapp] = odefun( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            Kij,gamma,bu,bl,Iamp,tau,T,s ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end

function [dY, Iapp] = odefun(t,Y,Kij,gamma,bu,bl,Iamp,tau,T,s)
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
    Inet = Kij * F(x);

    % system of equations
    dx = -gamma.*x + (x - bl).*(bu - x).*(Inet + Iapp);    

    % return
    dY = [dx];
end

% Sigmoid function
function y=F(x)
    y = 1./(1+exp(-x)) - 0.5;
end