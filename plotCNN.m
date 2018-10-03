figure
x=0:1:50;
y1=ringn1o10r3;
y2=ringn2o10r3;
y3=ringn3o10r3;
y4=ringn4o10r3;
y5=ringn5o10r3;
y6=ringn6o10r3;
y7=ringn7o10r3;
y8=ringn8o10r3;
plot(x,y1,'-s','LineWidth',4)
hold on
plot(x,y2,'--s','LineWidth',4)
hold on
plot(x,y3,':s','LineWidth',4)
hold on
plot(x,y4,'-.s','LineWidth',4)
hold on
plot(x,y5,'-x','LineWidth',4)
hold on
plot(x,y6,'--x','LineWidth',4)
hold on
plot(x,y7,':x','LineWidth',4)
hold on
plot(x,y8,'-.x','LineWidth',4)
hold on
legend({'Node 1','Node 2','Node 3','Node 4','Node 5','Node 6','Node 7','Node 8',},'FontSize',25)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
