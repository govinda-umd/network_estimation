function sys = Linear_2D_ODE(Kij,p)
    % LinearODE  Linear Ordinary Differential Equation in two variables
    %   Implements the system of linear ordinary differential equations
    %        x'(t) = a*x(t) + b*y(t)
    %        y'(t) = c*x(t) + d*y(t)
    %   for use with the Brain Dynamics Toolbox.
    %
    % Example 1: Using the Brain Dynamics graphical toolbox
    %   sys = LinearODE();      % construct the system struct
    %   gui = bdGUI(sys);       % open the Brain Dynamics GUI
    % 
    % Example 2: Using the Brain Dynamics command-line tools
    %   sys = LinearODE();                 % system struct
    %   sys = bdSetPar(sys,'a',1);         % parameter a=1
    %   sys = bdSetPar(sys,'b',-1);        % parameter b=-1
    %   sys = bdSetPar(sys,'c',10);        % parameter c=10
    %   sys = bdSetPar(sys,'d',-2);        % parameter d=-2
    %   sys = bdSetVar(sys,'x',rand);      % variable x=rand
    %   sys = bdSetVar(sys,'y',rand);      % variable y=rand
    %   tspan = [0 10];                    % soln time span
    %   sol = bdSolve(sys,tspan);          % call the solver
    %   tplot = 0:0.1:10;                  % plot time domain
    %   Y = bdEval(sol,tplot);             % extract solution
    %   plot(tplot,Y);                     % plot the result
    %   xlabel('time'); ylabel('x,y');
    %
    % Authors
    %   Stewart Heitmann (2017a,2018a,2019ab,2020a)

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
    
    n = size(Kij, 1);

    % Handle to our ODE function
    sys.odefun = @odefun;
    
    % ODE parameter definitions
    sys.pardef = [
        struct('name','Kij','value', Kij)
        struct('name','k', 'value', 1)
        struct('name','a', 'value', -1)
        struct('name','b', 'value',0.5)
        struct('name','c', 'value',0)
        struct('name','d', 'value',-2)
        struct('name','p', 'value',p)
        struct('name','s', 'value',zeros(n,1))
        ];
    
    % ODE variable definitions
    lim = p/2;
    sys.vardef = [
        struct('name','x', 'value',zeros(n,1), 'lim',[-lim lim])
        struct('name','y', 'value',zeros(n,1), 'lim',[-lim lim])
        ];

    % Latex (Equations) panel
    sys.panels.bdLatexPanel.title = 'Equations'; 
    sys.panels.bdLatexPanel.latex = { 
        '$\textbf{LinearODE}$'
        ''
        'System of linear ordinary differential equations'
        '{ }{ }{ } $\dot x(t) = a\,x(t) + b\,y(t)$'
        '{ }{ }{ } $\dot y(t) = c\,x(t) + d\,y(t)$'
        'where $a,b,c,d\;$ are scalar constants.'
        };

    % Time Portrait panel 
    sys.panels.bdTimePortrait = [];

    % Phase Portrait panel
    sys.panels.bdPhasePortrait.nullclines = 'on';
    sys.panels.bdPhasePortrait.vectorfield = 'on';
  
    % Solver panel
    sys.panels.bdSolverPanel = [];
    
    % Default time span (optional)
    sys.tspan = [0 20];

    % Specify the relevant ODE solvers (optional)
    sys.odesolver = {@ode45,@ode23,@odeEul};
    
    % ODE solver options (optional)
    sys.odeoption.RelTol = 1e-6;        % Relative Tolerance
%     sys.odeoption.Jacobian = @jacfun;   % Handle to Jacobian function 
    sys.odeoption.InitialStep = 0.01;   % Required by odeEul solver
end

% The ODE function.
% The variables Y and dYdt are both (2x1) vectors.
% The parameters a,b,c,d are scalars.
function dYdt = odefun(t,Y,Kij,k,a,b,c,d,p,s) 
    Y = reshape(Y, [], 2);
    x = Y(:, 1);
    y = Y(:, 2);

%     Inet = k .* f(Kij, (2*pi/p).*x);
    Inet = k .* f(Kij, x);
    
    dx = a.*x + b.*y + Inet;
    dy = c.*x + d.*y + s;
    dYdt = [dx; dy];
end

function [z] = f(Kij, x)
%     1.
    z = Kij * x;

%     2.
%     z = Kij * x;
%     z = 1 ./ (1 + exp(-z)) - 0.5;

%     3.
%     z = ones(size(Kij,1),1);
%     z = Kij * x - x .* Kij * z;

%     4. 
%     c_x = cos(x);
%     s_x = sin(x);
%     z = diag(c_x)*Kij*s_x - diag(s_x)*Kij*c_x;
end