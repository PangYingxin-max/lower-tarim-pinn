function out = VegetationWater
% Vegetation-water-storage system used in the Hopf verification.
% x = [P T W]; p = [a1 a2 a3 a4 b1 b2 b3 c1 c2 Win]
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = @der3;
out{8} = [];
out{9} = [];

function dydt = fun_eval(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
P = x(1); T = x(2); W = x(3);
dydt = [ a1*P*(1-P)*(1-T) - a3*T*P - a4*P + a2*W;
         b1*(1-P) - b2*W - b3*T;
         c1*(Win - W) - c2*P*W ];

function [tspan, y0, options] = init
handles = feval(VegetationWater);
y0 = [0.37; 0.53; 0.08];
options = odeset('Jacobian', handles(3), 'JacobianP', handles(4), ...
                 'Hessians', handles(5), 'HessiansP', handles(6));
tspan = [0 10];

function jac = jacobian(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
P = x(1); T = x(2); W = x(3);
jac = [ a1*(1-2*P)*(1-T) - a3*T - a4,  -a1*P*(1-P) - a3*P,  a2;
        -b1,                            -b3,                  -b2;
        -c2*W,                           0,                   -c1 - c2*P ];

function jacp = jacobianp(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
P = x(1); T = x(2); W = x(3);
jacp = [ P*(1-P)*(1-T),  W,  -T*P,  -P,   0,      0,      0,      0,        0,      0;
         0,               0,   0,     0,  1-P,   -W,     -T,      0,        0,      0;
         0,               0,   0,     0,   0,     0,      0,    Win-W,   -P*W,   c1 ];

function hess = hessians(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
P = x(1); T = x(2); W = x(3);
h1 = [ -2*a1*(1-T),  -a1*(1-2*P),  0;
       -a1*(1-2*P),   0,           0;
       0,             0,           0 ];
h2 = zeros(3);
h3 = [ 0, 0, -c2;  0, 0, 0;  -c2, 0, 0 ];
hess(:,:,1) = h1; hess(:,:,2) = h2; hess(:,:,3) = h3;

function hessp = hessiansp(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
P = x(1); T = x(2); W = x(3);
h = zeros(3,3,10);
h(1,1,1) = (1-2*P)*(1-T);  h(1,2,1) = -P*(1-P);
h(1,3,2) = 1;
h(1,1,3) = -T;             h(1,2,3) = -P;
h(1,1,4) = -1;
h(2,1,5) = -1;
h(2,3,6) = -1;
h(2,2,7) = -1;
h(3,3,8) = -1;
h(3,1,9) = -W;             h(3,3,9) = -P;
hessp = h;

function tens3 = der3(t, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, Win)
tens3 = zeros(3,3,3,3);
