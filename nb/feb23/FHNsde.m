% FHNsde Stochastic FitzHugh-Nagumo model of neural excitability
%    dV = (V - 1/3*V^3 - W + Iext)*dt + sigma1*dB1
%    dW = (V + a - b*W)*dt/tau + sigma2*dB2
%
% Example:
%   sys = FHNsde();         % Construct the system struct.
%   gui = bdGUI(sys);       % Open the Brain Dynamics GUI.
%
function sys = FHNsde()    
    % Handle to our SDE functions
    sys.sdeF = @sdeF;
    sys.sdeG = @sdeG;
    
    % Our SDE parameters
    sys.pardef = [
        struct('name','a',      'value',1,    'lim',[-5 5])
        struct('name','b',      'value',0.5,  'lim',[-1 1])
        struct('name','tau',    'value',10,   'lim',[1 20])
        struct('name','Iapp',   'value',1,    'lim',[0 5])
        struct('name','sigma1', 'value',0,    'lim',[0 1])
        struct('name','sigma2', 'value',0,    'lim',[0 1])
        ];
               
    % Our SDE variables        
    sys.vardef = [
        struct('name','V', 'value',rand, 'lim',[-3 3])
        struct('name','W', 'value',rand, 'lim',[-1 6])        
        ];
        
    % time span
    sys.tspan = [0 300];
    
    % SDE options
    sys.sdesolver = {@sdeEM,@sdeSH};
    sys.sdeoption.InitialStep = 0.01;
    sys.sdeoption.NoiseSources = 2;
    
    % Equations
    sys.panels.bdLatexPanel.title = 'Equations';
    sys.panels.bdLatexPanel.latex = {
        '$\textbf{FHNsde}$';
        '';
        'The stochastic FitzHugh-Nagumo (FHN) model of neural excitability';
        '{ }{ }{ } $dV = \big(V - \frac{1}{3}V^3 - W + I_{app} \big) dt + \sigma_1 d\xi_1$';
        '{ }{ }{ } $dW = \frac{1}{\tau}\big(V + a - b W \big) dt + \sigma_2 d\xi_2$';
        'where';
        '{ }{ }{ } $V(t)\;$ is the membrane voltage,';
        '{ }{ }{ } $W(t)\;$ is a recovery variable,';
        '{ }{ }{ } $\xi(t)\;$ is a Wiener process,';
        '{ }{ }{ } $a,b\;$ and $\tau\;$ are constants.';
        '';
        'References:';
        '{ }{ }{ } FitzHugh (1961) Impulses and physiological states in theoretical models of nerve membrane. Biophysical J. 1:445--466';
        '{ }{ }{ } Nagumo, Arimoto and Yoshizawa (1962) An active pulse transmission line simulating nerve axon. Proc. IRE. 50:2061--2070.';
        };
    
    % Display panels
    sys.panels.bdTimePortrait = [];
    sys.panels.bdPhasePortrait = [];
    sys.panels.bdSolverPanel = [];
end

% FitzHugh-Nagumo Stochastic Differential Equation (deterministic)
function F = sdeF(~,Y,a,b,tau,Iapp,sigma1,sigma2)
    % extract incoming variables
    V = Y(1);
    W = Y(2);
    
    % system of equations
    dV = V - 1/3*V^3 - W + Iapp;
    dW = (V + a - b*W) ./ tau;
    
    % return
    F = [ dV; dW];
end

% FitzHugh-Nagumo Stochastic Differential Equation (stochastic)
function G = sdeG(~,Y,a,b,tau,Iapp,sigma1,sigma2)
    G = [sigma1  0 ;
         0  sigma2 ]; 
end
