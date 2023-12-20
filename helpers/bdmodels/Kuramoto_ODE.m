% KuramotoNet - Network of Kuramoto Phase Oscillators
%   Constructs a Kuramoto network with n nodes.
%       theta_i' = omega_i + SUM_j Kij*sin(theta_i-theta_j)
%   where 
%       theta is an (nx1) vector of oscillator phases (in radians),
%       omega is an (nx1) vector of natural frequencies (cycles/sec)
%       Kij is an (nxn) matrix of connection weights,
%
% Example:
%   n = 20;                    % number of oscillators
%   Kij = ones(n);             % coupling matrix
%   sys = KuramotoNet(Kij);    % construct the system struct
%   gui = bdGUI(sys);          % open the Brain Dynamics GUI
%
% Authors
%   Stewart Heitmann (2016a,2017a,2018a,2018b,2020a)

% Copyright (C) 2016-2022 QIMR Berghofer Medical Research Institute
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
% 1. Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in
%    the documentation and/or other materials provided with the
%    distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
% COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
% ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
function sys = Kuramoto_ODE(Kij)
    % Determine the number of nodes from the size of the coupling matrix
    n = size(Kij,1);
    
    % Handle to our ODE function
    sys.odefun = @odefun;
    
    % ODE parameters
    sys.pardef = [ struct('name','Kij',   'value',Kij,       'lim',[-1,1]);
                   struct('name','beta',  'value',ones(n,1), 'lim',[-1,1]);
                   struct('name','k',     'value',zeros(2,1),'lim',[-10,10]);               
                   struct('name','A',     'value',ones(n,1), 'lim',[-pi,pi])
                   struct('name','B',     'value',0.75,      'lim',[0, 5])
                   struct('name','C',     'value',1.0,       'lim',[0, 5])
                   struct('name','d',     'value',1,         'lim',[-1,1]);
                   struct('name','omega', 'value',rand(n,1), 'lim',[-pi,pi])
                   struct('name','Iamp',  'value',1,         'lim',[0, 100])
                   struct('name','tau',   'value',10,        'lim',[0, 100])
                   struct('name','T',     'value',25,        'lim',[0, 100])
                   struct('name','s',     'value',zeros(n,1),'lim',[0, 1])
                 ];
               
    % ODE state variables
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[-pi pi]);
        struct('name','theta', 'value',2*pi*zeros(n,1), 'lim',[-pi pi]);
        ];
    
    % Time span
    sys.tspan = [0 100];
    sys.tstep = 0.1;

    % Relevant ODE solvers
    sys.odesolver = {@ode45,@ode23,@ode113,@odeEul};
    
    % ODE solver options              
    sys.odeoption.RelTol = 1e-6;
    sys.odeoption.MaxStep = 0.001;   
    sys.odeoption.InitialStep = 0.01;
                    
    % Latex (Equations) panel
    sys.panels.bdLatexPanel.title = 'Equations'; 
    sys.panels.bdLatexPanel.latex = {
        '$\textbf{KuramotoNet}$'
        ''
        'A generalised network of Kuramoto Oscillators'
        '{ }{ }{ } $\dot \theta_i = \omega_i + \frac{k}{n} \sum_j K_{ij} \sin(\theta_i - \theta_j)$'
        'where'
        '{ }{ }{ } $\theta_i$ is the phase of the $i^{th}$ oscillator (radians),'
        '{ }{ }{ } $\omega_i$ is its natural oscillation frequency (cycles/sec),'
        '{ }{ }{ } $K$ is the network connectivity matrix ($n$ x $n$),'
        '{ }{ }{ } $k$ is a scaling constant,'
        '{ }{ }{ } $i,j=1 \dots n,$'
        ['{ }{ }{ } $n{=}',num2str(n),'.$']
        ''
        'The Kuramoto order parameter ($R$) is a metric of phase synchronisation.'
        '{ }{ }{ } $R = \frac{1}{n} \| \sum_i \exp(\mathbf{i} \theta_i) \|$'
        'It corresponds to the radius of the centroid of the phases, as shown in'
        'the Auxiliary panel.'
        ''
        'References'
        '\qquad Kuramoto (1984) Chemical oscillations, waves and turbulence.'
        '\qquad Strogatz (2000) From Kuramoto to Crawford.'
        '\qquad Breakspear et al (2010) Generative models of cortical oscillations.'
        '\qquad Chapter 6.2 of the Handbook for the Brain Dynamics Toolbox (Version 2018b).'
        };
    
    % Time Portrait panel
    sys.panels.bdTimePortrait.title = 'Time Portrait';
    sys.panels.bdTimePortrait.modulo = 'on';
 
    % Phase Portrait panel
    sys.panels.bdPhasePortrait.title = 'Phase Portrait';
    sys.panels.bdPhasePortrait.modulo = 'on';
    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';

    % Auxiliary panel
    sys.panels.bdAuxiliary.title = 'Auxiliary';
    sys.panels.bdAuxiliary.auxfun = {@ts,@Stimulus}; 
    %{@centroid1,@centroid2,@KuramotoR,@ts,@Stimulus};
    
    % Solver panel
    sys.panels.bdSolverPanel.title = 'Solver';                
end

% ODE
function [dY, Iapp] = odefun(t,Y,Kij,beta,k,A,B,C,d,omega,Iamp,tau,T,s)
    Y = reshape(Y,[],2);
    x = Y(:, 1);
    theta = Y(:, 2);

    k_x = k(1);
    k_t = k(2);

    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end

%     c_x = cos(x);
%     s_x = sin(x);
%     Inet_x = k_x.*diag(c_x)*Kij*s_x - k_x.*diag(s_x)*Kij*c_x;
    Inet_x = k_x .* f(Kij * x);

    dx = -A.*x...
         +(B - C.*x.^2) .* (Inet_x + Iapp);
    
    if d == 1
        D = abs(x);
    elseif d == -1
        D = Iapp;
    else
        D = 1;
    end
    c_t = cos(theta);
    s_t = sin(theta);
    Inet_t = k_t.*diag(c_t)*Kij*s_t - k_t.*diag(s_t)*Kij*c_t;
    dtheta = omega + D .* (Inet_t);
    % D decides when to synchronize: when x != 0, or when Iapp != 0, or
    % everytime.

    dY = [dx; dtheta];
end

% Sigmoid function
function y=f(x)
    gamma = 1;
    y = 1./(1+exp(-x./gamma)) - 0.5;
    %y = tanh(x);
end

% Auxiliary function for plotting the signal.
function ts(ax,t,sol,Kij,beta,k,A,B,C,d,omega,Iamp,tau,T,s)
    n = size(Kij, 1);
    t = sol.x;
    x = sol.y(1:n, :);
    theta = sol.y(n+1:end, :);
    y = x + beta .* sin(theta);

    % plot the time series.
    axis(ax,'normal');
    plot(ax,t,y);
    
    % axis limits etc
    t0 = min(t([1 end]));
    t1 = max(t([1 end]));
    xlim(ax,[t0 t1]);
%     ylim(ax,[min(y(:)),max(y(:))]);
    xlabel(ax,'time');
    ylabel(ax,'y(t)');
    title(ax,{'Time series', 'y(t) = x(t) + sin(\theta(t))'});
    grid(ax, 'on');
end

function Stimulus(ax,t,sol,Kij,beta,k,A,B,C,d,omega,Iamp,tau,T,s)
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
            Kij,beta,k,A,B,C,d,omega,Iamp,tau,T,s ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end
