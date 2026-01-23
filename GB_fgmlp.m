%% ================== FG-MLP 点预测 + 区间预测（无子函数版, 归一化+反归一化版） ==================
clear
clc
close all

% 如果你想直接用工作区里的 X,y，可以把下面三行注释掉
data = readmatrix('ETTh1');
X    = data(:,1:end-1);
y    = data(:,end);

[T, d] = size(X);
if size(y,1) ~= T
    error('X 和 y 的时间长度不一致！');
end

% 保存原始 y，用于后面反归一化 & 计算指标
y_orig = y;

% ★ 全局 y 范围（原始尺度，用于 PINAW 归一化）
range_y_all = max(y_orig) - min(y_orig);
if range_y_all < 1e-8
    range_y_all = 1;
end

L           = 168;      % 窗口长度
h           = 24;       % 预测步长
K           = floor(L/2);   % FFT 频点数（最多 L/2）
hiddenSize  = 64;       % 每个频段支路隐层单元数
numEpochs   = 50;       % 训练轮数
batchSize   = 8;       % mini-batch 大小
lr          = 1e-3;     % 学习率
lambda_w    = 1e-4;     % 权重 L2 正则系数
lambda_g    = 1e-5;     % 门控 L1 正则系数
verbose     = true;     % 是否输出 loss

% 区间预测参数
confLevel   = 0.95;     % 目标置信水平
M_local     = 168;      % 局部残差尺度窗口长度（历史窗口）

% 模糊关联规则挖掘阈值
supp_min    = 0.02;     % 最小模糊支持度
conf_min    = 0.50;     % 最小模糊置信度

% γ 搜索范围（允许显著压缩，也允许略微放大）
gamma_min   = 0.4;
gamma_max   = 1.4;
gamma_num   = 161;       % 网格点个数

% η 的取值范围（允许适度放大/缩小）
eta_min     = 0.8;
eta_max     = 1.0;

% 允许训练集覆盖率比目标低多少（这里设 0，直接对齐 0.95）
deltaPICP_allow = 0.00;

% 训练/测试划分比例（按时间）
train_ratio = 0.8;

%% ---- 1.1 仅用训练段统计对 X、y 做标准化（防止信息泄露） ----
T_train_raw  = floor(T * train_ratio);      % 原始时间轴训练长度


% ★ X 归一化（训练段）
mu_X  = mean(X(1:T_train_raw,:), 1);
sig_X = std(X(1:T_train_raw,:), [], 1);
sig_X(sig_X < 1e-8) = 1;

X = bsxfun(@rdivide, bsxfun(@minus, X, mu_X), sig_X);


% ★ y 归一化（训练段）
mu_y  = mean(y_orig(1:T_train_raw));
sig_y = std(y_orig(1:T_train_raw));
if sig_y < 1e-8
    sig_y = 1;
end

y = (y_orig - mu_y) / sig_y;   % 后续所有建模都在归一化后的 y 上进行

%% ---------------------- 2. 构造滑动窗口样本 ----------------------
N = T - L - h + 1;
if N <= 0
    error('样本数 N <= 0，请检查 L 和 h 的设置。');
end

Xwin      = zeros(N, L, d);
ywin      = zeros(N, 1);   % 归一化后的标签
ywin_orig = zeros(N, 1);   % 原始尺度标签（用于最后计算指标）

for n = 1:N
    Xwin(n,:,:) = X(n : n+L-1, :);
    idx_target  = n + L - 1 + h;
    ywin(n)      = y(idx_target);        % 归一化
    ywin_orig(n) = y_orig(idx_target);   % 原始
end

%% ---------------------- 3. 频域特征提取 ----------------------
K = min(K, floor(L/2));        % 最多取 L/2 个频点
amps = zeros(N, K, d);         % 幅值: N x K x d

for n = 1:N
    Xn = squeeze(Xwin(n,:,:)); % L x d
    F  = fft(Xn);              % L x d
    A  = abs(F(1:K,:));        % 前 K 个频率幅值: K x d
    amps(n,:,:) = A;
end

% 展开为二维特征矩阵：N x (K*d)
Xfd = reshape(amps, N, K*d);

% 每个频域维度对应的原始特征编号
featureOfDim = repelem(1:d, K);    % 1 x (K*d)

% 按频率划分低/中/高频三段
k1 = floor(K/3);
k2 = floor(2*K/3);
Fdim = K*d;

idxLow  = false(1, Fdim);
idxMid  = false(1, Fdim);
idxHigh = false(1, Fdim);
for i = 1:d
    base = (i-1)*K;
    idxLow(base+1       : base+k1) = true;
    idxMid(base+k1+1    : base+k2) = true;
    idxHigh(base+k2+1   : base+K)  = true;
end

nLow  = sum(idxLow);
nMid  = sum(idxMid);
nHigh = sum(idxHigh);

%% ---------------------- 4. 按“窗口时间”划分训练 / 测试集 ----------------------
N_train = floor(N * train_ratio);
N_test  = N - N_train;

idx_train = 1:N_train;
idx_test  = N_train+1:N;

% 频域特征划分
Xfd_train  = Xfd(idx_train, :);
Xfd_test   = Xfd(idx_test,  :);   % ★ 这行是之前漏掉的

% 标签划分（归一化后的）
ywin_train = ywin(idx_train);
ywin_test  = ywin(idx_test);

% 标签划分（原始尺度，用于后面计算指标）
ywin_train_orig = ywin_orig(idx_train);
ywin_test_orig  = ywin_orig(idx_test);

% FFT 幅值划分
amps_train = amps(idx_train,:,:);
amps_test  = amps(idx_test,:,:);

%% ---- 4.1 频域特征 Xfd 再做一次基于训练集的标准化 ----
mu_fd  = mean(Xfd_train, 1);
sig_fd = std(Xfd_train, [], 1);
sig_fd(sig_fd < 1e-8) = 1;

Xfd_train = bsxfun(@rdivide, bsxfun(@minus, Xfd_train, mu_fd), sig_fd);
Xfd_test  = bsxfun(@rdivide, bsxfun(@minus, Xfd_test,  mu_fd), sig_fd);



%% ===== A. 粒球特征选择（简化版，只用于初始化 β） =====
% 1) 特征-目标相关性 r_i（在归一化空间中计算，等价于原始）
X_tr_raw = X(1:T_train_raw,:);    % 已归一化
y_tr_raw = y(1:T_train_raw);      % 已归一化

r_feat = zeros(1,d);
for i = 1:d
    xi = X_tr_raw(:,i);
    if std(xi) < 1e-8 || std(y_tr_raw) < 1e-8
        r_feat(i) = 0;
    else
        C = corrcoef(xi, y_tr_raw);
        r_feat(i) = C(1,2);
    end
end
r_abs = abs(r_feat);

% 2) 初始化粒球：一个包含全部特征的“大球”
balls = cell(0);
balls{1} = 1:d;

alpha_gb = zeros(1,d);          % 特征贡献累计
maxSplits   = 5 * d;
minBallSize = 3;

std_all   = std(r_abs);
hetThresh = 0.1 * std_all;

splitCount = 0;
while splitCount < maxSplits
    bestBallIdx = 0;
    bestHetero  = 0;
    for b = 1:numel(balls)
        members = balls{b};
        if numel(members) <= minBallSize
            continue;
        end
        rB = r_abs(members);
        hetero = std(rB);
        if hetero > bestHetero
            bestHetero = hetero;
            bestBallIdx = b;
        end
    end
    
    if bestBallIdx == 0 || bestHetero < hetThresh
        break;
    end
    
    members = balls{bestBallIdx};
    rB      = r_abs(members);
    
    [~,idxMin] = min(rB);
    [~,idxMax] = max(rB);
    seed1 = members(idxMin);
    seed2 = members(idxMax);
    
    B1 = [];
    B2 = [];
    for m = members
        if abs(r_feat(m) - r_feat(seed1)) <= abs(r_feat(m) - r_feat(seed2))
            B1(end+1) = m;
        else
            B2(end+1) = m;
        end
    end
    
    if isempty(B1) || isempty(B2)
        break;
    end
    
    meanParent = mean(r_abs(members));
    meanB1 = mean(r_abs(B1));
    meanB2 = mean(r_abs(B2));
    meanChild = max(meanB1, meanB2);
    dJ = max(0, meanChild - meanParent);
    
    if dJ > 0
        if meanB1 >= meanB2
            B_best = B1;
        else
            B_best = B2;
        end
        alpha_gb(B_best) = alpha_gb(B_best) + dJ/numel(B_best);
    end
    
    balls{bestBallIdx} = B1;
    balls{end+1}       = B2;
    
    splitCount = splitCount + 1;
end

alpha_gb = alpha_gb + r_abs;

alpha_min  = min(alpha_gb);
alpha_max  = max(alpha_gb);
alpha_norm = (alpha_gb - alpha_min) ./ (alpha_max - alpha_min + 1e-8);

eps_alpha  = 1e-3;
alpha_map  = alpha_norm*(1-2*eps_alpha) + eps_alpha;
beta       = log(alpha_map ./ (1 - alpha_map));    % logit 映射

%% ===== B. 初始化频域三支路 MLP 参数 =====
scale = 0.01;

W_low = scale * randn(hiddenSize, nLow);
b_low = zeros(hiddenSize, 1);

W_mid = scale * randn(hiddenSize, nMid);
b_mid = zeros(hiddenSize, 1);

W_high = scale * randn(hiddenSize, nHigh);
b_high = zeros(hiddenSize, 1);

W_out = scale * randn(1, 3*hiddenSize);
b_out = 0;

%% ===== C. 训练 FG-MLP 点预测（仅训练集） =====
lossEpoch = zeros(numEpochs, 1);

for epoch = 1:numEpochs
    idx_perm = randperm(N_train);
    epochLoss = 0;
    
    for startPos = 1:batchSize:N_train
        endPos = min(startPos+batchSize-1, N_train);
        batchIdx_local = idx_perm(startPos:endPos);
        
        Xb = Xfd_train(batchIdx_local, :);   % B x Fdim
        yb = ywin_train(batchIdx_local);     % B x 1 (归一化标签)
        B  = size(Xb,1);
        
        % ---------- 前向 ----------
        g     = 1 ./ (1 + exp(-beta));        % 1 x d
        g_dim = g(featureOfDim);              % 1 x Fdim
        Xg    = bsxfun(@times, Xb, g_dim);    % B x Fdim
        
        X_low  = Xg(:, idxLow);
        X_mid  = Xg(:, idxMid);
        X_high = Xg(:, idxHigh);
        
        A_low = X_low * W_low.' + repmat(b_low.', B, 1);
        Z_low = max(0, A_low);
        
        A_mid = X_mid * W_mid.' + repmat(b_mid.', B, 1);
        Z_mid = max(0, A_mid);
        
        A_high = X_high * W_high.' + repmat(b_high.', B, 1);
        Z_high = max(0, A_high);
        
        Z_cat = [Z_low, Z_mid, Z_high];
        y_hat_batch = Z_cat * W_out.' + b_out;
        
        % 损失：MSE + 权重 L2 + 门控 L1
        E       = y_hat_batch - yb;
        mseLoss = mean(E.^2);
        
        regW = norm(W_low,'fro')^2 + ...
               norm(W_mid,'fro')^2 + ...
               norm(W_high,'fro')^2 + ...
               norm(W_out,'fro')^2;
        regW = 0.5 * lambda_w * regW;
        regG = lambda_g * sum(abs(g));
        
        lossBatch = mseLoss + regW + regG;
        epochLoss = epochLoss + lossBatch * B;
        
        % ---------- 反向 ----------
        grad_y_hat = (2 / B) * E;
        
        grad_W_out = grad_y_hat.' * Z_cat + lambda_w * W_out;
        grad_b_out = sum(grad_y_hat);
        
        grad_Z_cat = grad_y_hat * W_out;
        H = hiddenSize;
        grad_Z_low  = grad_Z_cat(:, 1:H);
        grad_Z_mid  = grad_Z_cat(:, H+1:2*H);
        grad_Z_high = grad_Z_cat(:, 2*H+1:3*H);
        
        grad_A_low  = grad_Z_low  .* (A_low  > 0);
        grad_A_mid  = grad_Z_mid  .* (A_mid  > 0);
        grad_A_high = grad_Z_high .* (A_high > 0);
        
        grad_W_low = grad_A_low.' * X_low + lambda_w * W_low;
        grad_b_low = sum(grad_A_low, 1).';
        grad_X_low = grad_A_low * W_low;
        
        grad_W_mid = grad_A_mid.' * X_mid + lambda_w * W_mid;
        grad_b_mid = sum(grad_A_mid, 1).';
        grad_X_mid = grad_A_mid * W_mid;
        
        grad_W_high = grad_A_high.' * X_high + lambda_w * W_high;
        grad_b_high = sum(grad_A_high, 1).';
        grad_X_high = grad_A_high * W_high;
        
        grad_Xg = zeros(B, Fdim);
        grad_Xg(:, idxLow)  = grad_Xg(:, idxLow)  + grad_X_low;
        grad_Xg(:, idxMid)  = grad_Xg(:, idxMid)  + grad_X_mid;
        grad_Xg(:, idxHigh) = grad_Xg(:, idxHigh) + grad_X_high;
        
        grad_g_dim = sum(grad_Xg .* Xb, 1);
        grad_g = zeros(1, d);
        for i = 1:d
            idx_i = (featureOfDim == i);
            if any(idx_i)
                grad_g(i) = sum(grad_g_dim(idx_i));
            end
        end
        
        grad_g    = grad_g + lambda_g * sign(g);
        grad_beta = grad_g .* g .* (1 - g);
        
        % ---------- SGD 更新 ----------
        beta   = beta   - lr * grad_beta;
        W_low  = W_low  - lr * grad_W_low;
        b_low  = b_low  - lr * grad_b_low;
        W_mid  = W_mid  - lr * grad_W_mid;
        b_mid  = b_mid  - lr * grad_b_mid;
        W_high = W_high - lr * grad_W_high;
        b_high = b_high - lr * grad_b_high;
        W_out  = W_out  - lr * grad_W_out;
        b_out  = b_out  - lr * grad_b_out;
    end
    
    lossEpoch(epoch) = epochLoss / N_train;
    if verbose
        fprintf('Epoch %d/%d, train loss = %.6f\n', epoch, numEpochs, lossEpoch(epoch));
    end
end

%% ===== D. 训练集 & 测试集点预测（先在归一化空间算，再反归一化评估） =====
g_all     = 1 ./ (1 + exp(-beta));
g_dim_all = g_all(featureOfDim);

% ---- 训练集前向 ----
B_tr   = N_train;
Xb_tr  = Xfd_train;
Xg_tr  = bsxfun(@times, Xb_tr, g_dim_all);

X_low_tr  = Xg_tr(:, idxLow);
X_mid_tr  = Xg_tr(:, idxMid);
X_high_tr = Xg_tr(:, idxHigh);

A_low_tr  = X_low_tr  * W_low.'  + repmat(b_low.',  B_tr, 1);
A_mid_tr  = X_mid_tr  * W_mid.'  + repmat(b_mid.',  B_tr, 1);
A_high_tr = X_high_tr * W_high.' + repmat(b_high.', B_tr, 1);

Z_low_tr  = max(0, A_low_tr);
Z_mid_tr  = max(0, A_mid_tr);
Z_high_tr = max(0, A_high_tr);

Z_cat_tr      = [Z_low_tr, Z_mid_tr, Z_high_tr];
y_point_train = Z_cat_tr * W_out.' + b_out;   % 归一化预测

% ---- 测试集前向 ----
B_te   = N_test;
Xb_te  = Xfd_test;
Xg_te  = bsxfun(@times, Xb_te, g_dim_all);

X_low_te  = Xg_te(:, idxLow);
X_mid_te  = Xg_te(:, idxMid);
X_high_te = Xg_te(:, idxHigh);

A_low_te  = X_low_te  * W_low.'  + repmat(b_low.',  B_te, 1);
A_mid_te  = X_mid_te  * W_mid.'  + repmat(b_mid.',  B_te, 1);
A_high_te = X_high_te * W_high.' + repmat(b_high.', B_te, 1);

Z_low_te  = max(0, A_low_te);
Z_mid_te  = max(0, A_mid_te);
Z_high_te = max(0, A_high_te);

Z_cat_te = [Z_low_te, Z_mid_te, Z_high_te];
y_point  = Z_cat_te * W_out.' + b_out;       % 测试集归一化预测

% ★ 在原始尺度上计算 RMSE
y_point_train_orig = y_point_train * sig_y + mu_y;
y_point_orig       = y_point * sig_y + mu_y;

rmse_train = sqrt(mean((y_point_train_orig - ywin_train_orig).^2));
rmse_point = sqrt(mean((y_point_orig       - ywin_test_orig ).^2));

fprintf('训练集点预测 RMSE (原始尺度) = %.4f\n', rmse_train);
fprintf('测试集点预测 RMSE (原始尺度) = %.4f\n', rmse_point);

%% ===== 3.3.1 预测区间构建：训练阶段基线区间（在归一化空间中构建） =====
N_tr  = N_train;
res_tr = ywin_train - y_point_train;   % 残差 e_t（归一化）

% (1) 局部残差尺度（历史窗口：t-M+1..t）
s_local_tr = zeros(N_tr,1);
for t = 1:N_tr
    i1 = max(1, t - M_local + 1);
    e_seg = res_tr(i1:t);
    s_local_tr(t) = sqrt(mean(e_seg.^2));
end
s_local_tr(s_local_tr < 1e-8) = 1e-8;

% (2) 标准化残差 & 全局分位数
z_tr     = res_tr ./ s_local_tr;
z_abs_tr = abs(z_tr);
z_sorted = sort(z_abs_tr);
idx_q    = max(1, ceil(confLevel * numel(z_sorted)));
q        = z_sorted(idx_q);     % 标准化残差的经验分位数（归一化空间）

% 基线半宽度 & 区间（训练集，归一化尺度）
w_base_tr  = q * s_local_tr;
W_base_tr  = 2 * w_base_tr;

inside_base_tr = (ywin_train >= y_point_train - w_base_tr) & ...
                 (ywin_train <= y_point_train + w_base_tr);
PICP_base_tr = mean(inside_base_tr);
fprintf('训练集基线区间覆盖率（仅基线） PICP_base_tr = %.4f\n', PICP_base_tr);

% 归一化偏差（真实值相对基线区间的位置，用于期望调整标签）
delta_tr = (ywin_train - y_point_train) ./ (w_base_tr + 1e-8);

%% ===== 3.3.2 不确定性特征构造：高频能量占比 + 基线宽度相对值 =====
% 高频能量占比（使用门控加权后的高频能量 / 总能量）
Eh_tr   = zeros(N_tr,1);
Etot_tr = zeros(N_tr,1);
for n = 1:N_tr
    A_n = squeeze(amps_train(n,:,:));   % K x d
    A2  = A_n.^2;
    G2  = g_all.^2;
    G2_mat = repmat(G2, K, 1);
    A2_eff = A2 .* G2_mat;
    Eh_tr(n)   = sum(sum(A2_eff(k2+1:K,:)));
    Etot_tr(n) = sum(sum(A2_eff));
end
hf_ratio_tr = Eh_tr ./ (Etot_tr + 1e-8);    % 相对高频能量占比（0~1）

% ★ 基线区间相对宽度（相对于全局 y 波动范围，使用原始尺度宽度）
global_scale = range_y_all;
if global_scale < 1e-8
    global_scale = 1;
end
W_base_tr_orig = W_base_tr * sig_y;      % 把基线宽度从归一化尺度映射回原始尺度
width_rel_tr = W_base_tr_orig / (global_scale + 1e-8);  % 区间宽度 / 总波动范围

% 三角形隶属函数
triMF = @(x,a,b,c) max(min((x-a)./(b-a+1e-8), (c-x)./(c-b+1e-8)), 0);

% 高频能量占比的分位数
hf_min = min(hf_ratio_tr);
hf_max = max(hf_ratio_tr);
hf_q25 = quantile(hf_ratio_tr, 0.25);
hf_q50 = quantile(hf_ratio_tr, 0.50);
hf_q75 = quantile(hf_ratio_tr, 0.75);

hf_params = zeros(3,3);  % L/M/H 的 [a,b,c]
hf_params(1,:) = [hf_min, hf_q25, hf_q50];  % L
hf_params(2,:) = [hf_q25, hf_q50, hf_q75];  % M
hf_params(3,:) = [hf_q50, hf_q75, hf_max];  % H
if hf_q25 == hf_min, hf_params(1,2) = (hf_min+hf_q50)/2; end
if hf_q75 == hf_max, hf_params(3,2) = (hf_q50+hf_max)/2; end

% 宽度相对指标的分位数
wd_min = min(width_rel_tr);
wd_max = max(width_rel_tr);
wd_q25 = quantile(width_rel_tr, 0.25);
wd_q50 = quantile(width_rel_tr, 0.50);
wd_q75 = quantile(width_rel_tr, 0.75);

wd_params = zeros(3,3);          % N(窄), A(适中), W(宽)
wd_params(1,:) = [wd_min, wd_q25, wd_q50]; % N
wd_params(2,:) = [wd_q25, wd_q50, wd_q75]; % A
wd_params(3,:) = [wd_q50, wd_q75, wd_max]; % W
if wd_q25 == wd_min, wd_params(1,2) = (wd_min+wd_q50)/2; end
if wd_q75 == wd_max, wd_params(3,2) = (wd_q50+wd_max)/2; end

% 训练样本的输入端模糊隶属度
mu_hf_tr = zeros(N_tr,3);   % 列：L/M/H
mu_wd_tr = zeros(N_tr,3);   % 列：N/A/W

for t = 1:N_tr
    x_hf = hf_ratio_tr(t);
    x_wd = width_rel_tr(t);
    for k = 1:3
        mu_hf_tr(t,k) = triMF(x_hf, hf_params(k,1), hf_params(k,2), hf_params(k,3));
        mu_wd_tr(t,k) = triMF(x_wd, wd_params(k,1), wd_params(k,2), wd_params(k,3));
    end
end

%% ===== 3.3.3 模糊关联规则挖掘 =====
% (1) 期望调整标签 label_tr：根据 delta_tr 粗略划分 SS,S,K,E,EE
label_tr = zeros(N_tr,1);   % 1~5: SS,S,K,E,EE

for t = 1:N_tr
    dt = delta_tr(t);
    % 更保守的划分：0.3 / 0.8
    if dt >= 0
        if abs(dt) <= 0.3
            label_tr(t) = 3;     % K
        elseif abs(dt) <= 0.8
            label_tr(t) = 4;     % E
        else
            label_tr(t) = 5;     % EE
        end
    else
        if abs(dt) <= 0.3
            label_tr(t) = 3;     % K
        elseif abs(dt) <= 0.8
            label_tr(t) = 2;     % S
        else
            label_tr(t) = 1;     % SS
        end
    end
end

% 输出端模糊隶属度（这里用 crisp：标签项为1，其它为0）
mu_out_tr = zeros(N_tr,5);   % 列：SS,S,K,E,EE
for t = 1:N_tr
    lab = label_tr(t);
    if lab >=1 && lab <=5
        mu_out_tr(t, lab) = 1;
    end
end

% (2) 模糊规则度量
rule_hf   = [];
rule_wd   = [];
rule_out  = [];
rule_supp = [];
rule_conf = [];

for idx_hf = 1:3
    for idx_wd = 1:3
        % 前件模糊支持度
        mu_X = min(mu_hf_tr(:,idx_hf), mu_wd_tr(:,idx_wd));   % N_tr x 1
        supp_X = mean(mu_X);
        if supp_X < 1e-6
            continue;
        end
        
        for idx_o = 1:5
            mu_R = min(mu_X, mu_out_tr(:,idx_o));
            supp_R = mean(mu_R);
            conf_R = supp_R / (supp_X + 1e-8);
            
            if supp_R >= supp_min && conf_R >= conf_min
                rule_hf(end+1)   = idx_hf;
                rule_wd(end+1)   = idx_wd;
                rule_out(end+1)  = idx_o;
                rule_supp(end+1) = supp_R;
                rule_conf(end+1) = conf_R;
            end
        end
    end
end

numRules = numel(rule_hf);
if numRules == 0
    % 若未挖掘到满足阈值的规则，则给一个“保持基线”的默认规则
    rule_hf   = 2;    % hf 为 M
    rule_wd   = 2;    % wd 为 A
    rule_out  = 3;    % K（保持）
    rule_supp = 1;
    rule_conf = 1;
    numRules  = 1;
end

fprintf('挖掘到的模糊规则数: %d\n', numRules);

%% ===== 3.3.4 在线推理与区间调整 =====
% 输出变量“区间调整因子”的语言项 {SS,S,K,E,EE} 的代表值
% 低不确定性：强/弱缩窄；高不确定性：弱/强放大
v_SS = 0.75;
v_S  = 0.90;
v_K  = 1.00;
v_E  = 1.15;
v_EE = 1.30;
v_vec = [v_SS, v_S, v_K, v_E, v_EE];

%% ---- 4.1 测试集基线区间（仍在归一化空间构建） ----
N_te  = N_test;
res_te = ywin_test - y_point;   % 残差（归一化）

s_local_te = zeros(N_te,1);
for t = 1:N_te
    i1 = max(1, t - M_local + 1);
    e_seg = res_te(i1:t);
    s_local_te(t) = sqrt(mean(e_seg.^2));
end
s_local_te(s_local_te < 1e-8) = 1e-8;

w_base_te = q * s_local_te;
W_base_te = 2 * w_base_te;

% 测试集基线区间的 PICP（归一化空间下与原始一致）
inside_base_te = (ywin_test >= y_point - w_base_te) & ...
                 (ywin_test <= y_point + w_base_te);
PICP_base_te = mean(inside_base_te);

% ★ 基线 PINAW 使用原始尺度宽度
PINAW_base_te = mean(2*w_base_te * sig_y) / (range_y_all + 1e-8);

fprintf('测试集基线区间（不含模糊） PICP_base_te = %.4f, PINAW_base_te = %.4f\n', ...
    PICP_base_te, PINAW_base_te);

%% ---- 4.2 测试集不确定性特征 & 模糊化 ----
Eh_te   = zeros(N_te,1);
Etot_te = zeros(N_te,1);
for n = 1:N_te
    A_n = squeeze(amps_test(n,:,:));   % K x d
    A2  = A_n.^2;
    G2  = g_all.^2;
    G2_mat = repmat(G2, K, 1);
    A2_eff = A2 .* G2_mat;
    Eh_te(n)   = sum(sum(A2_eff(k2+1:K,:)));
    Etot_te(n) = sum(sum(A2_eff));
end
hf_ratio_te = Eh_te ./ (Etot_te + 1e-8);

W_base_te_orig = W_base_te * sig_y;
width_rel_te   = W_base_te_orig / (global_scale + 1e-8);

mu_hf_te = zeros(N_te,3);
mu_wd_te = zeros(N_te,3);
for t = 1:N_te
    x_hf = hf_ratio_te(t);
    x_wd = width_rel_te(t);
    for k = 1:3
        mu_hf_te(t,k) = triMF(x_hf, hf_params(k,1), hf_params(k,2), hf_params(k,3));
        mu_wd_te(t,k) = triMF(x_wd, wd_params(k,1), wd_params(k,2), wd_params(k,3));
    end
end

%% ---- 4.3 训练集上的 η_train（用于γ标定） ----
eta_tr = ones(N_tr,1);

for t = 1:N_tr
    nume = 0;
    deno = 0;
    for r = 1:numRules
        h_idx = rule_hf(r);
        w_idx = rule_wd(r);
        o_idx = rule_out(r);
        
        mu_X = min(mu_hf_tr(t,h_idx), mu_wd_tr(t,w_idx));  % 前件激活度
        phi_r = mu_X;
        
        if phi_r > 1e-8
            v_r = v_vec(o_idx);
            nume = nume + phi_r * v_r;
            deno = deno + phi_r;
        end
    end
    if deno > 1e-8
        eta_tr(t) = nume / deno;
    else
        eta_tr(t) = 1.0;   % 没有规则激活则保持基线
    end
    % 将 η 限制在 [eta_min, eta_max]
    eta_tr(t) = max(eta_min, min(eta_tr(t), eta_max));
end

%% ---- 4.4 γ标定：在训练集上搜索最优 γ，使 PICP ≥ (confLevel - deltaPICP_allow) 且 PINAW 最小 ----
gammas = linspace(gamma_min, gamma_max, gamma_num);
PICP_tr_gamma  = zeros(gamma_num,1);
PINAW_tr_gamma = zeros(gamma_num,1);

range_y_tr = range_y_all;   % 用原始尺度范围做归一化

for ig = 1:gamma_num
    gamma = gammas(ig);
    w_tr_gamma = gamma * eta_tr .* w_base_tr;   % 仍是归一化尺度半宽
    
    y_L_tr_g = y_point_train - w_tr_gamma;
    y_U_tr_g = y_point_train + w_tr_gamma;
    
    inside_g = (ywin_train >= y_L_tr_g) & (ywin_train <= y_U_tr_g);
    PICP_tr_gamma(ig)  = mean(inside_g);
    
    % PINAW 使用原始尺度宽度
    width_tr_gamma_orig = (y_U_tr_g - y_L_tr_g) * sig_y;
    PINAW_tr_gamma(ig)  = mean(width_tr_gamma_orig) / (range_y_tr + 1e-8);
end

PICP_target_gamma = max(0, confLevel - deltaPICP_allow);

best_idx       = 1;
best_PINAW     = inf;
found_feasible = false;

for ig = 1:gamma_num
    if PICP_tr_gamma(ig) >= PICP_target_gamma
        if ~found_feasible || PINAW_tr_gamma(ig) < best_PINAW
            best_PINAW = PINAW_tr_gamma(ig);
            best_idx   = ig;
            found_feasible = true;
        end
    end
end

if ~found_feasible
    % 如果没有任何 γ 能达到放宽后的覆盖率要求，则仍然选择 PICP 最大的那一个
    [~, best_idx] = max(PICP_tr_gamma);
end

gamma_eta = gammas(best_idx);
fprintf('γ标定结果: gamma_eta = %.4f, 对应训练集 PICP = %.4f, PINAW = %.4f (目标下限 = %.4f)\n', ...
    gamma_eta, PICP_tr_gamma(best_idx), PINAW_tr_gamma(best_idx), PICP_target_gamma);

%% ---- 4.5 测试集上的 η_te & 最终区间（最后一步反归一化） ----
eta_te = ones(N_te,1);

for t = 1:N_te
    nume = 0;
    deno = 0;
    for r = 1:numRules
        h_idx = rule_hf(r);
        w_idx = rule_wd(r);
        o_idx = rule_out(r);
        
        mu_X = min(mu_hf_te(t,h_idx), mu_wd_te(t,w_idx));  % 前件激活度
        phi_r = mu_X;
        
        if phi_r > 1e-8
            v_r = v_vec(o_idx);
            nume = nume + phi_r * v_r;
            deno = deno + phi_r;
        end
    end
    if deno > 1e-8
        eta_te(t) = nume / deno;
    else
        eta_te(t) = 1.0;
    end
    eta_te(t) = max(eta_min, min(eta_te(t), eta_max));
end

% 最终半宽（归一化尺度）
w_final_norm = gamma_eta * eta_te .* w_base_te;

% ★ 反归一化：得到原始尺度上的预测和区间
y_point_orig = y_point * sig_y + mu_y;
w_final      = w_final_norm * sig_y;

y_L = y_point_orig - w_final;
y_U = y_point_orig + w_final;

%% ===== 测试集区间评价：PICP & PINAW（全部在原始尺度） =====
inside = (ywin_test_orig >= y_L) & (ywin_test_orig <= y_U);
PICP   = mean(inside);

PINAW  = mean(y_U - y_L) / (range_y_all + 1e-8);

fprintf('测试集区间指标: PICP = %.4f, PINAW = %.4f (目标置信水平 = %.2f)\n', ...
    PICP, PINAW, confLevel);

%% ================== E. 区间预测结果可视化（测试集） ==================
% 测试集每个样本对应原始时间索引（目标点在原序列中的位置）
t_test = (idx_test(:) + L - 1 + h);   % N_te x 1

% 1) 预测区间 + 点预测 + 真实值（全测试集）
figure('Name','Test Prediction Interval (Full)','Color','w');
hold on; box on; grid on;

% 阴影区间
fill([t_test; flipud(t_test)], [y_L; flipud(y_U)], ...
     [0.85 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.6);

% 上下界（可选：更清晰）
plot(t_test, y_L, '-', 'LineWidth', 0.8);
plot(t_test, y_U, '-', 'LineWidth', 0.8);

% 点预测 & 真实值
plot(t_test, y_point_orig, '-', 'LineWidth', 1.5);


legend({'95% PI band','Lower','Upper','Point pred','True'}, 'Location','best');
xlabel('Time index (original series)');
ylabel('y (original scale)');
title(sprintf('Test Prediction Interval (PICP=%.3f, PINAW=%.3f)', PICP, PINAW));


