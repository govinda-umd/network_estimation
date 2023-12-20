function sys = coupled_oscillators_ODE(Kij)
    % Determine the number of nodes from the size of the coupling matrix
    n = size(Kij, 1);

    % Handle to ODE function
    sys.odefun = @odefun;
    
    % ODE parameters
    sys.pardef = [
        struct('name','Kij',   'value',Kij,       'lim',[-1,1])
        struct('name','a',     'value',1,         'lim',[-5, 5])
        struct('name','b',     'value',5,         'lim',[-5, 5])
        struct('name','gamma', 'value',1,         'lim',[-5, 5])
        struct('name','k',     'value',2,         'lim',[-5, 5])
        struct('name','Iamp',  'value',1,         'lim',[0, 100])
        struct('name','tau',   'value',10,        'lim',[0, 100])
        struct('name','T',     'value',25,        'lim',[0, 100])
        struct('name','s',     'value',zeros(n,1),'lim',[0, 1])
    ];
    
    % ODE variables 
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[-5, 5])
        struct('name','w', 'value',zeros(n,1), 'lim',[-5, 5])
    ];
    
    % time span
    sys.tspan = [0 100];
    sys.tstep = 0.001;

    % ode options
    sys.odesolver = {@ode45,@ode23,@ode23s,@odeEul};
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
    sys.panels.bdAuxiliary.auxfun = {@Stimulus,@Inet};

    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';

end

function Stimulus(ax,t,sol,Kij,a,b,gamma,k,Iamp,tau,T,s)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,A, ...: model parameters 
    % Reconstruct the stimulus used by odefun
    Iapp = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,iapp,~,~] = odefun( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            Kij,a,b,gamma,k,Iamp,tau,T,s ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end

function Inet(ax,t,sol,Kij,a,b,gamma,k,Iamp,tau,T,s)
    % ax: current axis
    % t: current time step
    % sol: solution returned by the solver (ode45)
    % Kij,A, ...: model parameters 
    % Reconstruct the stimulus used by odefun
    Inet = zeros(size(s,1),size(sol.x,2));
    for idx = 1:numel(sol.x)
        [~,~,inet,~] = odefun( ...
            sol.x(idx), ...
            sol.y(:,idx), ...
            Kij,a,b,gamma,k,Iamp,tau,T,s ...
            );
        Inet(:, idx) = inet;
    end
    
    %plot the stimulus
    plot(ax,sol.x,Inet')
    xlabel(ax,'time');
    ylabel(ax,'Inet');
    title(ax,'Network stimulus');
    legend(ax);
end

% deterministic part
function [F, Iapp, Inet, Itot] = odefun(t,Y,Kij,a,b,gamma,k,Iamp,tau,T,s)
    Y = reshape(Y, [], 2);
    x = Y(:, 1);
    w = Y(:, 2);

    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end
    
    Inet = k .* nonlinear_func(Kij, x);

    Itot = Inet + Iapp;

    % equations
    dx = w;
    dw = -a.*w - b.*x + gamma .* Itot;

    F = [dx; dw];
end

function [y] = nonlinear_func(Kij,x)
%     1. difference
    z = ones(size(Kij,1),1);
    y = Kij * x - x .* Kij * z;

%     2. sigmoid over linear
%     y = Kij * x;
%     y = 1 ./ (1 + exp(-y)) - 0.5;

%     3. sigmoid over difference
%     z = ones(size(Kij,1),1);
%     y = Kij * x - x .* Kij * z; 
%     y = 1 ./ (1 + exp(-y)) - 0.5;   

%     4. kuramoto sinusoid
    c_x = cos(x);
    s_x = sin(x);
    y = diag(c_x)*Kij*s_x - diag(s_x)*Kij*c_x;

end