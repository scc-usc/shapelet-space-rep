dd = csvread("flu_sims.csv");
%%
plot(dd', 'Color', [0 0 1 0.2]); hold on;
plot(mean(dd, 1, 'omitnan'), 'o', 'Color', 'r')

%% Shapelet definitions
A = [1 1.5 3 7.5; ...
    1 2 3 4; ...
    1,2,2,1; ...
    4,3,2,1; ...
    0 0 0 0];

A = [1 2 3 4;
    1 2 2 1;
   1 2 4 8;
    0 0 0 0];

% A = [1 2 3 4 5;
%     1 2 3 2 1;
%     1 2 4 8 16;
%     0 0 0 0 0];



d = size(A, 1); w = size(A, 2);
%% Find slopes that are definite increase
ns = size(dd, 1); T = size(dd, 2);
slope_time = T;
slope_thres = zeros(ns, 1);
for cid = 1:ns
    slope_thres(cid) = max(movmean(abs(diff(dd(cid, 1:slope_time))), [0 d-1]));
end
slope_thres(:) = median(slope_thres, 'omitnan');
%% Compute shapelet space representation at all points in time for all time-series

[all_reps] = shape_ts_transform(dd, A, slope_thres);

%% Plot shapelets
cid = randi(size(dd, 1), 1);% cid = 107;
tiledlayout(2, 1); 

nexttile; plot(dd(cid, :)); xlim([1,size(all_reps, 2)]);
xlabel('Time'); ylabel('Value');
title('Time-series');
nexttile; 
imagesc(squeeze(all_reps(cid, :, :))'); xlim([1,size(all_reps, 2)]);
colorbar('eastoutside');
xlabel('Time'); ylabel('Shapelet dimensions');
ax = gca;
ax.YTickLabel = {'Inc', 'Peak', 'Surge', 'Flat'};
title('Shapelet space visualization');
%%
win = 30;

%% Find similarity matrix (this will be used later for clustering in case we want to only create ensemble of one cluster)
tic;
sim_mat = nan(ns, ns); 
%par
for ii=1:ns
    for jj=1:ii
        sim_mat(ii, jj) = dtw_cons_md(squeeze(all_reps(ii, :, :))', squeeze(all_reps(jj, :, :))', (win), 'euc');
    end
end
toc
