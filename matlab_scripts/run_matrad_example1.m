%% run_matrad_example1.m
% Headless matRad example1 via Octave.
% Optimizer: Octave sqp() — replaces fmincon/IPOPT (MATLAB-only).

graphics_toolkit('gnuplot');
set(0, 'DefaultFigureVisible', 'off');

MATRAD_ROOT = 'C:\Users\jkim20\Desktop\projects\tps\matRad';
addpath(MATRAD_ROOT);
cd(MATRAD_ROOT);
matRad_rc;
matRad_cfg = MatRad_Config.instance();
matRad_cfg.logLevel = 1;

disp('=== matRad example1 (headless, Octave) ===');

%% Phantom
ctDim        = [200, 200, 100];
ctResolution = [2, 2, 3];
builder = matRad_PhantomBuilder(ctDim, ctResolution, 1);
builder.addSphericalTarget('Volume1', 20, 'objectives', ...
    struct(DoseObjectives.matRad_SquaredDeviation(800, 45)), 'HU', 0);
builder.addBoxOAR('Volume2', [60,30,60], 'offset', [0 -15 0], 'objectives', ...
    struct(DoseObjectives.matRad_SquaredOverdosing(400, 0)));
builder.addBoxOAR('Volume3', [60,30,60], 'offset', [0  15 0], 'objectives', ...
    struct(DoseObjectives.matRad_SquaredOverdosing(10, 0)));
[ct, cst] = builder.getctcst();
disp('Phantom built.');

%% Plan
pln.radiationMode           = 'photons';
pln.machine                 = 'Generic';
pln.bioModel                = 'none';
pln.multScen                = 'nomScen';
pln.numOfFractions          = 30;
pln.propStf.gantryAngles    = 0:70:355;
pln.propStf.couchAngles     = zeros(size(pln.propStf.gantryAngles));
pln.propStf.bixelWidth      = 5;
pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst, ct, 0);
pln.propOpt.runDAO          = 0;
pln.propOpt.runSequencing   = 0;
pln.propDoseCalc.doseGrid.resolution.x = 2;
pln.propDoseCalc.doseGrid.resolution.y = 2;
pln.propDoseCalc.doseGrid.resolution.z = 3;

%% STF + dij
disp('Generating STF...');
stf = matRad_generateStf(ct, cst, pln);
fprintf('  %d beams, %d bixels\n', numel(stf), sum([stf.totalNumOfBixels]));

disp('Computing dose influence matrix...');
dij = matRad_calcDoseInfluence(ct, cst, stf, pln);
fprintf('  dij: %d voxels x %d bixels\n', dij.doseGrid.numOfVoxels, dij.totalNumOfBixels);

%% Uniform fluence forward dose
disp('Computing forward dose with uniform fluence (w = 1.0)...');

D   = dij.physicalDose{1};   % sparse (Nvox x Nbixel)
nB  = dij.totalNumOfBixels;
nFx = pln.numOfFractions;

w_opt = ones(nB, 1);  % uniform: all bixels = 1.0

%% Result
dose_fx  = full(D * w_opt);               % Gy/fx
dose_tot = dose_fx * nFx;                 % total Gy

fprintf('\n=== Result ===\n');
fprintf('  max dose (total):    %.2f Gy\n', max(dose_tot));
fprintf('  max dose (per fx):   %.4f Gy\n', max(dose_fx));
fprintf('  weights: %d  nnz: %d  max: %.4f\n', nB, sum(w_opt > 1e-4), max(w_opt));

%% Export to .mat for Python/OpenTPS
outdir  = 'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\_data\example1_octave';
if ~exist(outdir, 'dir'), mkdir(outdir); end
outfile = fullfile(outdir, 'result.mat');

% Reshape dose to 3D (Ny,Nx,Nz) on dose grid
dose_3d = reshape(dose_tot, dij.doseGrid.dimensions);  % [Gy total]

% CT cube (Ny,Nx,Nz)
ct_hu = ct.cubeHU{1};

% Grid info
ct_origin  = [ct.x(1), ct.y(1), ct.z(1)];
ct_spacing = [ct.resolution.x, ct.resolution.y, ct.resolution.z];
dg_origin  = [dij.doseGrid.x(1), dij.doseGrid.y(1), dij.doseGrid.z(1)];
dg_spacing = [dij.doseGrid.resolution.x, dij.doseGrid.resolution.y, dij.doseGrid.resolution.z];

save('-v7', outfile, 'dose_3d', 'ct_hu', 'ct_origin', 'ct_spacing', 'dg_origin', 'dg_spacing', 'w_opt');
fprintf('Saved result to: %s\n', outfile);

disp('Done.');
exit(0);  % clean exit before FLTK cleanup can segfault
