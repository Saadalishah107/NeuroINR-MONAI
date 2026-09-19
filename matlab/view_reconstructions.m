function view_reconstructions(mat_path)
% VIEW_RECONSTRUCTIONS  Load and display a NeuroINR-MONAI .mat export.
%
% Usage (MATLAB or Octave):
%   >> cd matlab
%   >> view_reconstructions('../outputs/matlab_exports/resolution_experiment.mat')
%
% Displays the original image alongside every reconstruction/target pair
% found in the file (fields named recon_* / target_*), with PSNR/SSIM in
% each subplot title.

    if nargin < 1
        error('Usage: view_reconstructions(mat_path)');
    end

    S = load(mat_path);
    fn = fieldnames(S);
    recon_fields = fn(startsWith(fn, 'recon_'));

    n = numel(recon_fields);
    figure('Name', 'NeuroINR-MONAI reconstructions', 'Position', [100 100 300*(n+1) 350]);

    subplot(1, n+1, 1);
    if isfield(S, 'original_image')
        imagesc(S.original_image); axis image off; colormap gray;
        title('Original');
    end

    for i = 1:n
        key = erase(recon_fields{i}, 'recon_');
        recon = S.(recon_fields{i});
        psnr_field = ['psnr_' key];
        ssim_field = ['ssim_' key];

        subplot(1, n+1, i+1);
        imagesc(recon); axis image off; colormap gray;

        ttl = strrep(key, '_', ' ');
        if isfield(S, psnr_field) && isfield(S, ssim_field)
            ttl = sprintf('%s\nPSNR=%.1fdB SSIM=%.3f', ttl, S.(psnr_field), S.(ssim_field));
        end
        title(ttl, 'Interpreter', 'none');
    end
end
