function run_coco_hopf()
% Independent COCO continuation of the three-state Hopf-capable system.
root = 'C:\Users\ACER\PycharmProjects\pythonProject';
addpath(genpath(fullfile(root, 'thirdparty', 'coco', 'cocotools-master', 'toolbox')));
cd(fullfile(root, 'matcont'));

p = struct('a1', 10.2, 'a2', 0.647, 'a3', 0.299, 'a4', 2.99, ...
           'b1', 0.113, 'b2', 0.435, 'b3', 0.0679, 'c1', 0.00852, 'c2', 0.177);
u0 = [0.368617230557656; 0.5330018516118152; 0.0808170740748106; 0.699705593654581];

prob = coco_prob();
prob = coco_set(prob, 'cont', 'NAdapt', 5, 'h_max', 0.02, 'PtMX', 200, 'ItMX', 20);
prob = coco_add_func(prob, 'eq', @(o, d, x) eqres(o, d, x, p), [], 'zero', 'x0', u0);
prob = coco_add_pars(prob, '', [1 2 3 4], {'P', 'T', 'W', 'WIN'});
coco(prob, 'run_coco', [], 1, {'WIN', 'P', 'T', 'W'});

bd = coco_bd_read('run_coco');
WIN = coco_bd_col(bd, 'WIN');
P = coco_bd_col(bd, 'P');
T = coco_bd_col(bd, 'T');
W = coco_bd_col(bd, 'W');

n = numel(WIN);
rho = zeros(n, 1);
for k = 1:n
    J = jac3(P(k), T(k), W(k), p);
    ev = eig(J);
    cplx = ev(abs(imag(ev)) > 1e-9);
    if isempty(cplx)
        rho(k) = NaN;
    else
        rho(k) = max(real(cplx));
    end
end
[WIN, order] = sort(WIN);
P = P(order); T = T(order); W = W(order); rho = rho(order);

Win_scan = linspace(min(WIN), max(WIN), 251);
rho_scan = arrayfun(@(w) realpair(w, p), Win_scan);
hopf = [];
for i = 1:numel(Win_scan) - 1
    if rho_scan(i) * rho_scan(i + 1) < 0
        h = bisect(@(w) realpair(w, p), Win_scan(i), Win_scan(i + 1));
        if isfinite(h) && (isempty(hopf) || all(abs(hopf - h) > 1e-6))
            hopf(end + 1) = h; %#ok<AGROW>
        end
    end
end
if isempty(hopf)
    hopf = NaN;
end

fb = fopen(fullfile(root, 'results_hopf', 'coco_branch.csv'), 'w');
fprintf(fb, 'WIN,P,T,W,rho\n');
for k = 1:n
    fprintf(fb, '%.12f,%.12f,%.12f,%.12f,%.6e\n', WIN(k), P(k), T(k), W(k), rho(k));
end
fclose(fb);
fid = fopen(fullfile(root, 'results_hopf', 'coco_hopf.csv'), 'w');
fprintf(fid, 'tool,WIN,n_points\n');
fprintf(fid, 'COCO,%.12f,%d\n', hopf(1), n);
fclose(fid);
fprintf('COCO points=%d hopf=%.12f\n', n, hopf(1));
end

function [data y] = eqres(prob, data, x, p) %#ok<INUSD>
P = x(1); T = x(2); W = x(3); WIN = x(4);
y = [p.a1 * P * (1 - P) * (1 - T) - p.a3 * T * P - p.a4 * P + p.a2 * W; ...
     p.b1 * (1 - P) - p.b2 * W - p.b3 * T; ...
     p.c1 * (WIN - W) - p.c2 * P * W];
end

function J = jac3(P, T, W, p)
J = [p.a1 * (1 - 2 * P) * (1 - T) - p.a3 * T - p.a4, ...
     -p.a1 * P * (1 - P) - p.a3 * P, p.a2; ...
     -p.b1, -p.b3, -p.b2; ...
     -p.c2 * W, 0, -p.c1 - p.c2 * P];
end

function y = realpair(WIN, p)
[P, T, W] = upper_state(WIN, p);
ev = eig(jac3(P, T, W, p));
cplx = ev(abs(imag(ev)) > 1e-9);
if isempty(cplx)
    y = 1.0;
else
    y = max(real(cplx));
end
end

function y = scalareq(P, WIN, p)
W = p.c1 * WIN / (p.c1 + p.c2 * P);
T = (p.b1 * (1 - P) - p.b2 * W) / p.b3;
y = p.a1 * P * (1 - P) * (1 - T) - p.a3 * T * P - p.a4 * P + p.a2 * W;
end

function [P, T, W] = upper_state(WIN, p)
Pgrid = linspace(1e-4, 1 - 1e-4, 2000);
g = arrayfun(@(q) scalareq(q, WIN, p), Pgrid);
idx = find(g(1:end-1) .* g(2:end) < 0);
P = NaN;
for k = numel(idx):-1:1
    Pk = bisect(@(q) scalareq(q, WIN, p), Pgrid(idx(k)), Pgrid(idx(k) + 1));
    if isfinite(Pk) && Pk > 1e-8
        P = Pk;
        break;
    end
end
W = p.c1 * WIN / (p.c1 + p.c2 * P);
T = (p.b1 * (1 - P) - p.b2 * W) / p.b3;
end

function x = bisect(f, a, b, n)
if nargin < 4, n = 200; end
fa = f(a); fb = f(b);
if ~isfinite(fa) || ~isfinite(fb) || sign(fa) == sign(fb)
    x = NaN;
    return;
end
for k = 1:n
    m = 0.5 * (a + b);
    fm = f(m);
    if ~isfinite(fm)
        x = m;
        return;
    end
    if sign(fm) == sign(fa)
        a = m; fa = fm;
    else
        b = m;
    end
end
x = 0.5 * (a + b);
end
