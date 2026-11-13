function run_matcont_hopf()
root = 'C:\Users\ACER\PycharmProjects\pythonProject';
addpath(genpath(fullfile(root, 'thirdparty', 'matcont', 'MatCont7p6')));
addpath(fullfile(root, 'matcont'));
global cds eds
cds = []; eds = [];

p = [10.2 0.647 0.299 2.99 0.113 0.435 0.0679 0.00852 0.177 0.7460];
x0 = [0.4782; 0.4313; 0.0682];

[x0v, v0] = init_EP_EP(@VegetationWater, x0, p, 10);
opt = contset();
opt = contset(opt, 'MaxNumPoints', 160);
opt = contset(opt, 'MaxStepSize', 0.002);
opt = contset(opt, 'MinStepSize', 1e-6);
opt = contset(opt, 'Backward', 1);
opt = contset(opt, 'SymDerivative', 3);
opt = contset(opt, 'SymDerivativeP', 2);
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'TestFunctions', 1);
[x, v, s, h, f] = cont(@equilibrium, x0v, v0, opt);

fprintf('MATCONT_POINTS %d\n', size(x, 2));
fprintf('MATCONT_LAST_WIN %.10f\n', x(end, end));
fprintf('MATCONT_FIRST_WIN %.10f\n', x(end, 1));
for k = 1:numel(s)
    fprintf('MATCONT_SINGULARITY %s index=%d win=%.10f P=%.10f T=%.10f W=%.10f\n', ...
        s(k).label, s(k).index, x(end, s(k).index), x(1, s(k).index), ...
        x(2, s(k).index), x(3, s(k).index));
end

out = fullfile(root, 'results_hopf', 'matcont_branch.csv');
writematrix(x.', out);
fprintf('MATCONT_CSV %s\n', out);

fid = fopen(fullfile(root, 'results_hopf', 'matcont_hopf.csv'), 'w');
fprintf(fid, 'label,win,P,T,W\n');
for k = 1:numel(s)
    if strcmp(strtrim(s(k).label), 'H')
        fprintf(fid, 'H,%.12f,%.12f,%.12f,%.12f\n', x(end,s(k).index), ...
            x(1,s(k).index), x(2,s(k).index), x(3,s(k).index));
    end
end
fclose(fid);
end
