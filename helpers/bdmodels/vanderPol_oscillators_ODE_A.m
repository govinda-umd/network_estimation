function sys = vanderPol_oscillators_ODE_A(Kij)
    % VanDerPolOscillators  System of N coupled van der Pol ocillators
    %   Implements a set of n coupled van der Pol equation
    %        Ui' = Vi
    %        Vi' = a*(1-Ui^2)*Vi - Ui - b*Kij*Ui
    %   where i=1..n and Kij is an nxn coupling matrix
    %
    % Example 1: Using the Brain Dynamics GUI
    %   n = 20;                             % number of nodes
    %   Kij = circshift(eye(n),1) + ...     % nearest-neighbour coupling
    %         circshift(eye(n),-1);
    %   sys = VanDerPolOscillators(Kij);    % construct the system struct
    %   gui = bdGUI(sys);                   % open the Brain Dynamics GUI
    % 
    % Example 2: Using the Brain Dynamics command-line solver
    %   n = 20;                                         % number of nodes
    %   Kij = circshift(eye(n),1) + ...                 % nearest-neighbour
    %         circshift(eye(n),-1);                     % coupling
    %   sys = VanDerPolOscillators(Kij);                % system struct
    %   sys = bdSetPar(sys,'a',1);                      % set 'a' parameter
    %   sys = bdSetPar(sys,'b',1.3);                    % set 'b' parameter
    %   sys = bdSetVar(sys,'U',rand(n,1));              % set 'y1' variable
    %   sys = bdSetVar(sys,'V',rand(n,1));              % set 'y2' variable
    %   sys.tspan = [0 10];                             % set time domain
    %   sol = bdSolve(sys);                             % solve
    %   tplot = 0:0.1:10;                               % plot time domain
    %   U = bdEval(sol,tplot,1:n);                      % U solution
    %   V = bdEval(sol,tplot,(1:n)+n);                  % V solution
    %   plot(tplot,U,'b', tplot,V,'r');                 % plot the result
    %   xlabel('time'); ylabel('U (blue), V (red)');    % axis labels
    %
    % Authors
    %   Stewart Heitmann (2016a,2017a,2018a,2020a)

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

    % determine the number of nodes from Kij
    n = size(Kij,1);

    % Handle to our ODE function
    sys.odefun = @odefun;
    
    % ODE parameters
    sys.pardef = [ 
        struct('name','Kij', 'value',Kij,  'lim',[0 2])
        struct('name','a',   'value',0.5,  'lim',[-1 2])
        struct('name','b',   'value',0.05,  'lim',[0 1])
        struct('name','c',   'value',0.05,  'lim',[0 1])
        struct('name','Iamp', 'value',1, 'lim',[0, 10])
        struct('name','tau', 'value',10, 'lim',[0, 10])
        struct('name','T', 'value',25, 'lim',[0, 10])
        struct('name','s', 'value',zeros(n,1), 'lim',[0, 1])
        ];
    
    % ODE variables           
    sys.vardef = [ 
        struct('name','x',   'value',0.05*ones(n,1), 'lim',[-2.5 2.5])
        struct('name','w',   'value',0.00*ones(n,1), 'lim',[-2.5 2.5])
        ];
               
    % Default time span
    sys.tspan = [0 100];
    sys.tstep = 0.1;
              
    % Specify ODE solvers and default options
    sys.odeoption.RelTol = 1e-6;                % ODE solver options
    %sys.odeoption.AbsTol = 1e-6;               
    sys.odeoption.InitialStep = 0.1;
    
    % Include the Latex (Equations) panel in the GUI
    sys.panels.bdLatexPanel.title = 'Equations'; 
    sys.panels.bdLatexPanel.latex = {
        '$\textbf{van der Pol Oscillators}$'
        ''
        'A network of coupled van der Pol oscillators'
        '{ }{ }{ } $\dot x_i = w_i - a\, g(x_i)$'
        '{ }{ }{ } $\dot w_i = -x_i - b\,\sum_j K_{ij}\,x_j - c\, I_i(t)$'
        'where'
        '{ }{ }{ } $g(z) = \frac{z^3}{3} - z$,'
        '{ }{ }{ } $x_i(t)\;$ and $w_i(t)\;$ are the dynamic variables ($n\;$ x $1$),'
        '{ }{ }{ } $K_{ij}\;$ is the connectivity matrix ($n\;$ x $n$),'
        '{ }{ }{ } $a\;$ and $b\;$ are scalar constants,'
        '{ }{ }{ } $i,j{=}1 \dots n$.'
        ''
        'Notes'
        ['{ }{ }{ } 1. This simulation has $n{=}',num2str(n),'$.']
        '{ }{ }{ } 2. Oscillations occur for $a>0$.'
        '{ }{ }{ } 3. Network coupling is scaled by $b$.'
        };
    
    % Include the Time Portrait panel in the GUI
    sys.panels.bdTimePortrait = [];
 
    % Include the Phase Portrait panel in the GUI
    sys.panels.bdPhasePortrait = [];

    % Include the Space-Time panel in the GUI
    sys.panels.bdSpaceTime = [];

    % Include the Solver panel in the GUI
    sys.panels.bdSolverPanel = [];   
    
    % Include the Auxiliary panel in the GUI
    sys.panels.bdAuxiliary.auxfun = {@Stimulus};
end

function Stimulus(ax,t,sol,Kij,a,b,c,Iamp,tau,T,s)
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
            Kij,a,b,c,Iamp,tau,T,s ...
            );
        Iapp(:, idx) = iapp;
    end
    
    %plot the stimulus
    stairs(ax,sol.x,Iapp')
    xlabel(ax,'time');
    ylabel(ax,'Iapp');
    title(ax,'Stimulus');
end

% The ODE function.
function [dYdt, Iapp] = odefun(t,Y,Kij,a,b,c,Iamp,tau,T,s)  
    % extract incoming [U,V] values from Y
    Y = reshape(Y,[],2);                % reshape Y to two colums
    x = Y(:,1);                         % 1st column contains x
    w = Y(:,2);                         % 2nd column contains w

    % square pulse
    if (mod(t, T) <= tau)
        Iapp = Iamp .* s;
    else
        Iapp = 0.0;
    end
    
    % Coupled van der Pol equations (in vector form)
    dx = w - a.*g(x);
    dw = -x - b.*Kij*x - c.*Iapp;
    
    % return a column vector
    dYdt=[dx; dw];
end

function y = g(x)
    y = (x.^3) ./ 3 - x;
end
   
