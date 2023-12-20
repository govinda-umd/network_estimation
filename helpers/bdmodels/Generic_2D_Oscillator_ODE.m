function sys = Generic_2D_Oscillator_ODE(Kij)
    % Determine the number of nodes from the size of the coupling matrix
    n = size(Kij, 1);

    % Handle to ODE function
    sys.odefun = @odefun;
    
    % ODE parameters
    sys.pardef = [
        struct('name','Kij',   'value',Kij,       'lim',[-1,1])
        struct('name','mu',    'value',1,         'lim',[1, 5])
        struct('name','a',     'value',0,         'lim',[-5, 5])
        struct('name','b',     'value',-10,       'lim',[-20, 15])
        struct('name','c',     'value',0,         'lim',[-10, 10])
        struct('name','beta',  'value',1,         'lim',[-5, 5])
        struct('name','e',     'value',1,         'lim',[-5, 5])
        struct('name','f',     'value',1,         'lim',[-5, 5])
        struct('name','g',     'value',0,         'lim',[-5, 5])
        struct('name','alpha', 'value',1,         'lim',[-5, 5])
        struct('name','gamma', 'value',1,         'lim',[-5, 5])
        struct('name','k',     'value',5,         'lim',[-5, 5])
        struct('name','Iamp',  'value',1,         'lim',[0, 100])
        struct('name','tau',   'value',10,        'lim',[0, 100])
        struct('name','T',     'value',25,        'lim',[0, 100])
        struct('name','s',     'value',rand(n,1), 'lim',[0, 1])
    ];
    
    % ODE variables 
    sys.vardef = [
        struct('name','v', 'value',zeros(n,1), 'lim',[-5, 5])
        struct('name','w', 'value',zeros(n,1), 'lim',[-25, 25])
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

function Stimulus(ax,t,sol,Kij,mu,a,b,c,beta,e,f,g,alpha,gamma,k,Iamp,tau,T,s)
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
            Kij,mu,a,b,c,beta,e,f,g,alpha,gamma,k,Iamp,tau,T,s ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end

function Inet(ax,t,sol,Kij,mu,a,b,c,beta,e,f,g,alpha,gamma,k,Iamp,tau,T,s)
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
            Kij,mu,a,b,c,beta,e,f,g,alpha,gamma,k,Iamp,tau,T,s ...
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
function [F, Iapp, Inet, Itot] = odefun(t,Y,Kij,mu,a,b,c,beta,e,f,g,alpha,gamma,k,Iamp,tau,T,s)
    Y = reshape(Y, [], 2);
    v = Y(:, 1);
    w = Y(:, 2);

    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end

    % equations
    v0 = -(2*e)/(6*f);
    w0 = -1*(-f.*v0.^3 - e.*v0.^2 + g.*v0);
    v_ = v - v0;
    w_ = w - w0./alpha;

    Inet = k .* nonlinear_func(Kij, w_);
    Itot = Inet + Iapp;

    dv = -f.*v_.^3 + e.*v_.^2 + g.*v_ + alpha.*w_ - gamma.*Itot;
    dv = mu.*dv;
    dw = c.*v.^2 + b.*v - beta.*w + a;
    dw = dw./mu;

    F = [dv; dw];
end

function [y] = nonlinear_func(Kij,x)
    z = ones(size(Kij,1),1);
    y = Kij * x - x .* Kij * z;
end