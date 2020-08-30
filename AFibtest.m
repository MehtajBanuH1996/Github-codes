clc; 
clear all;
close all;
fileName = '.mat'
load(fileName);
ecg = (val);
f_s=250;
N=length(ecg);
t=[0:N-1]/f_s; %time period(total sample/Fs )
figure
plot(t,ecg); title('Raw ECG Data plotting ')             
xlabel('time')
ylabel('amplitude')
w=50/(250/2);
bw=w;
[num,den]=iirnotch(w,bw); % notch filter implementation 
ecg_notch=filter(num,den,ecg);
figure,
N1=length(ecg_notch);
t1=[0:N1-1]/f_s;
plot(t1,ecg_notch,'r'); title('Filtered ECG signal ')             
xlabel('time')
ylabel('amplitude')
[e,f]=wavedec(ecg_notch,10,'db6');% Wavelet implementation
g=wrcoef('a',e,f,'db6',8); 
ecg_wave=ecg_notch-g; % subtracting 10th level aproximation signal
                       %from original signal                  
ecg_smooth=smooth(ecg_wave); % using average filter to remove glitches
                             %to increase the performance of peak detection 
N1=length(ecg_smooth);
t1=(0:N1-1)/f_s;
figure,
plot(t1,ecg_smooth,'r')
set(gca,'XTick',[], 'YTick',[])
imageName = strcat(erase(fileName,".mat"),'.png')
saveas(gcf,strcat('./AFib_ECG1/',imageName))