figure
x=0:1:50;
y1=alln2o10r1;
y2=ringn2o10r1;
% y3=alln2o0r1;

plot(x,y1,'-','LineWidth',4)
hold on
plot(x,y2,'--','LineWidth',4)
hold on
% plot(x,y3,':','LineWidth',4)
% hold on
legend({'All-to-all topology','Ring topology'},'FontSize',25)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])