%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for many bootstrap replications or many samples. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

mex BootstrapMeanGPU.cpp -lcudart -lcurand -lcublas -lStatistics -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Statistics/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Statistics/Statistics

Nsamples = 1000000;
Nbootstraps = 100;

data = 100*randn(Nsamples,1) + 332;

means = zeros(Nbootstraps,1);

start = clock;   
for b = 1:Nbootstraps
    indices = randi(Nsamples,Nsamples,1);
    newdata = data(indices);
    %means(b) = mean(newdata);    
    means(b) = sum(newdata)/Nsamples;    
end
CPU_time = etime(clock,start)

means2 = zeros(Nbootstraps,1);

for b = 1:Nbootstraps
    indices = randi(Nsamples,Nsamples,1);
    newdata = data(indices);
    means2(b) = sum(newdata)/Nsamples;    
end

[means_GPU,GPU_time] = BootstrapMeanGPU(data,Nbootstraps); 

[mean(means) mean(means2) mean(means_GPU)]

[std(means) std(means2) std(means_GPU)]

figure(1)
hist(means,320:0.25:345)
axis([320 345 0 0.1*Nbootstraps])
figure(2)
hist(means_GPU,320:0.25:345)
axis([320 345 0 0.1*Nbootstraps])

speedup = CPU_time / GPU_time*1000
