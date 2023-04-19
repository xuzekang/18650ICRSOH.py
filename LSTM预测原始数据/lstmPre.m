%% ����ǰ����
clc
clear
%% ���·��
addpath(genpath(pwd))
%% ��������
% ��������
Capacity_162 = csvread('capacity_discharge_162.csv');
Capacity_265 = csvread('capacity_discharge_265.csv');
Capacity_404 = csvread('capacity_discharge_404.csv');
Capacity_488 = csvread('capacity_discharge_488.csv');
Capacity_503 = csvread('capacity_discharge_503.csv');
Capacity_856 = csvread('capacity_discharge_856.csv');
Capacity_877 = csvread('capacity_discharge_877.csv');
%%����״̬
% ����״̬
S0H_162 = Capacity_162/max(Capacity_162);
S0H_265 = Capacity_265/max(Capacity_265);
S0H_404 = Capacity_404/max(Capacity_404);
S0H_488 = Capacity_488/max(Capacity_488);

% LSTMԤ������
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
%%Ԥ������ת����SOH
lstm_pre_162_50 = lstm_pre_162_50/max(Capacity_162);
lstm_pre_162_150 = lstm_pre_162_150/max(Capacity_162);

lstm_pre_265_50 = lstm_pre_265_50/max(Capacity_265);
lstm_pre_265_150 = lstm_pre_265_150/max(Capacity_265);

lstm_pre_404_50 = lstm_pre_404_50/max(Capacity_404);
lstm_pre_404_150 = lstm_pre_404_150/max(Capacity_404);

lstm_pre_488_50 = lstm_pre_488_50/max(Capacity_488);
lstm_pre_488_150 = lstm_pre_488_150/max(Capacity_488);
%% ��ŵ�ѭ����
Cycle_162 = (1:length(Capacity_162))';
Cycle_265 = (1:length(Capacity_265))';
Cycle_404 = (1:length(Capacity_404))';
Cycle_488 = (1:length(Capacity_488))';
Cycle_503 = (1:length(Capacity_503))';
Cycle_856 = (1:length(Capacity_856))';
Cycle_877 = (1:length(Capacity_877))';

%% ��ͼ
%% Ԥ�����
SP1 = 50;
SP2 = 150;
% 18650��Ԫ﮵��
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
legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

% 21700��Ԫ﮵��
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
legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);
% 18650����﮵��
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
legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

% 18650����﮵��
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
legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50','\fontname{Times new roman}SP=150' )
xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
ylabel('\fontname{Times new roman}SOH','FontSize',7);
set(gcf,'color','white');
set(gca,'FontName','Times New Roman', 'FontSize',7);
set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% 18650������﮵��
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
% legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
% ylabel('\fontname{����}����\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% 32700������﮵��
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
% legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
% ylabel('\fontname{����}����\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);

%% �������������
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
% legend('\fontname{����}��ʵֵ', '\fontname{Times new roman}SP=50', '\fontname{Times new roman}SP=150')
% xlabel('\fontname{����}��ŵ�ѭ����','FontSize',7);
% ylabel('\fontname{����}����\fontname{Times new roman}/Ah','FontSize',7);
% set(gcf,'color','white');
% set(gca,'FontName','Times New Roman', 'FontSize',7);
% set(gca, 'LooseInset', [0,0,0.04,0.03]);