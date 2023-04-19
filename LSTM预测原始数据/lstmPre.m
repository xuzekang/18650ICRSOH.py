%% 运行前操作
clc
clear
%% 添加路径
addpath(genpath(pwd))
%% 导入数据
% 容量数据
Capacity_162 = csvread('capacity_discharge_162.csv');
Capacity_265 = csvread('capacity_discharge_265.csv');
Capacity_404 = csvread('capacity_discharge_404.csv');
Capacity_488 = csvread('capacity_discharge_488.csv');
Capacity_503 = csvread('capacity_discharge_503.csv');
Capacity_856 = csvread('capacity_discharge_856.csv');
Capacity_877 = csvread('capacity_discharge_877.csv');
%%健康状态
% 健康状态
S0H_162 = Capacity_162/max(Capacity_162);
S0H_265 = Capacity_265/max(Capacity_265);
S0H_404 = Capacity_404/max(Capacity_404);
S0H_488 = Capacity_488/max(Capacity_488);

% LSTM预测数据
lstm_pre_162_50 = csvread('lstm_pre_162_50.csv');
lstm_pre_162_150 = csvread('lstm_pre_162_150.csv');
lstm_pre_265_50 = csvread('lstm_pre_265_50.csv');
lstm_pre_265_150 = csvread('lstm_pre_265_150.csv');
lstm_pre_404_50 = csvread('lstm_pre_404_50.csv');
lstm_pre_404_150 = csvread('lstm_pre_404_150.csv');
lstm_pre_488_50 = csvread('lstm_pre_488_50.csv');
lstm_pre_488_150 = csvread('lstm_pre_488_150.csv');
lstm_pre_503_50 = csvread('lstm_pre_503_50.csv');
lstm_pre_503_150 = csvread('lstm_pre_503_150.csv');
lstm_pre_877_50 = csvread('lstm_pre_877_50.csv');
lstm_pre_877_150 = csvread('lstm_pre_877_150.csv');
lstm_pre_856_50 = csvread('lstm_pre_856_50.csv');
lstm_pre_856_150 = csvread('lstm_pre_856_150.csv');
%%预测数据转换成SOH
lstm_pre_162_50 = lstm_pre_162_50/max(Capacity_162);
lstm_pre_162_150 = lstm_pre_162_150/max(Capacity_162);

lstm_pre_265_50 = lstm_pre_265_50/max(Capacity_265);
lstm_pre_265_150 = lstm_pre_265_150/max(Capacity_265);

lstm_pre_404_50 = lstm_pre_404_50/max(Capacity_404);
lstm_pre_404_150 = lstm_pre_404_150/max(Capacity_404);

lstm_pre_488_50 = lstm_pre_488_50/max(Capacity_488);
lstm_pre_488_150 = lstm_pre_488_150/max(Capacity_488);
%% 充放电循环数
Cycle_162 = (1:length(Capacity_162))';
Cycle_265 = (1:length(Capacity_265))';
Cycle_404 = (1:length(Capacity_404))';
Cycle_488 = (1:length(Capacity_488))';
Cycle_503 = (1:length(Capacity_503))';
Cycle_856 = (1:length(Capacity_856))';
Cycle_877 = (1:length(Capacity_877))';

%% 绘图
%% 预测起点
SP1 = 50;
SP2 = 150;
% 18650三元锂电池
figure
set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
plot(Cycle_162, S0H_162, 'b-', 'LineWidth', 1.5)
hold on
plot(Cycle_162(SP1:end,:), lstm_pre_162_50, 'r-.','LineWidth', 1.5)
hold on
plot(Cycle_162(SP2:end,:), lstm_pre_162_150, 'g:', 'LineWidth', 1.5)
hold on
plot([SP1 SP1], ylim, 'k', 'LineWidth', 1)
hold on
plot([SP2 SP2], ylim, 'k', 'LineWidth', 1)
legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{宋体}充放电循环数','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

% 21700三元锂电池
figure
set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
plot(Cycle_265, S0H_265, 'b-', 'LineWidth', 1.5)
hold on
plot(Cycle_265(SP1:end,:), lstm_pre_265_50, 'r-.','LineWidth', 1.5)
hold on
plot(Cycle_265(SP2:end,:), lstm_pre_265_150, 'g:', 'LineWidth', 1.5)
hold on
plot([SP1 SP1], ylim, 'k', 'LineWidth', 1)
hold on
plot([SP2 SP2], ylim, 'k', 'LineWidth', 1)
legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{宋体}充放电循环数','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);
% 18650锰酸锂电池
figure
set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
plot(Cycle_404, S0H_404, 'b-', 'LineWidth', 1.5)
hold on
plot(Cycle_404(SP1:end,:), lstm_pre_404_50, 'r-.','LineWidth', 1.5)
hold on
plot(Cycle_404(SP2:end,:), lstm_pre_404_150, 'g:', 'LineWidth', 1.5)
hold on
plot([SP1 SP1], ylim, 'k', 'LineWidth', 1)
hold on
plot([SP2 SP2], ylim, 'k', 'LineWidth', 1)
legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{宋体}充放电循环数','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

% 18650钴酸锂电池
figure
set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
plot(Cycle_488, S0H_488, 'b-', 'LineWidth', 1.5)
hold on
plot(Cycle_488(SP1:end,:), lstm_pre_488_50, 'r-.','LineWidth', 1.5)
hold on
plot(Cycle_488(SP2:end,:), lstm_pre_488_150, 'g:', 'LineWidth', 1.5)
hold on
plot([SP1 SP1], ylim, 'k', 'LineWidth', 1)
hold on
plot([SP2 SP2], ylim, 'k', 'LineWidth', 1)
legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{宋体}充放电循环数','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% 18650磷酸铁锂电池
% figure
% set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
% plot(Cycle_503, Capacity_503, 'b-', 'LineWidth', 1.5)
% hold on
% plot(Cycle_503(SP1:end,:), lstm_pre_503_50, 'r-.','LineWidth', 1.5)
% hold on
% plot(Cycle_503(SP2:end,:), lstm_pre_503_150, 'g:', 'LineWidth', 1.5)
% hold on
% plot([SP1 SP1], [min(ylim) 1.8], 'k', 'LineWidth', 1)
% hold on
% plot([SP2 SP2], [min(ylim) 1.77], 'k', 'LineWidth', 1)
% legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{宋体}充放电循环数','FontSize',7);
% ylabel('\fontname{宋体}容量\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% 32700磷酸铁锂电池
% figure
% set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
% plot(Cycle_877, Capacity_877, 'b-', 'LineWidth', 1.5)
% hold on
% plot(Cycle_877(SP1:end,:), lstm_pre_877_50, 'r-.','LineWidth', 1.5)
% hold on
% plot(Cycle_877(SP2:end,:), lstm_pre_877_150, 'g:', 'LineWidth', 1.5)
% hold on
% plot([SP1 SP1], [min(ylim) 5.9], 'k', 'LineWidth', 1)
% hold on
% plot([SP2 SP2], [min(ylim) 5.85], 'k', 'LineWidth', 1)
% legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{宋体}充放电循环数','FontSize',7);
% ylabel('\fontname{宋体}容量\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% 磷酸铁锂软包电池
% figure
% set(gcf, 'unit', 'centimeters', 'position', [10 10 7 7*0.618]);
% plot(Cycle_856, Capacity_856, 'b-', 'LineWidth', 1.5)
% hold on
% plot(Cycle_856(SP1:end,:), lstm_pre_856_50, 'r-.','LineWidth', 1.5)
% hold on
% plot(Cycle_856(SP2:end,:), lstm_pre_856_150, 'g:', 'LineWidth', 1.5)
% hold on
% plot([SP1 SP1], [min(ylim) 9.09], 'k', 'LineWidth', 1)
% hold on
% plot([SP2 SP2], [min(ylim) 9.055], 'k', 'LineWidth', 1)
% legend('\fontname{宋体}真实值', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{宋体}充放电循环数','FontSize',7);
% ylabel('\fontname{宋体}容量\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);