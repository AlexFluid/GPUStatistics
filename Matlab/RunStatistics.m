%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large images / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

mex Statistics.cpp -lcudart -lcurand -lStatistics -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Statistics/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Statistics/Statistics

Nsamples = 1000;
Nbootstraps = 100000;

data = 100*randn(Nsamples,1) + 332;
data_single = single(data);

means = zeros(Nbootstraps,1,'single');

start = clock;   
for b = 1:Nbootstraps
    indices = randi(Nsamples,Nsamples,1);
    means(b) = mean(data_single(indices) );    
end
CPU_time = etime(clock,start)

means = double(means);
[means_GPU,GPU_time] = Statistics(data,Nbootstraps);

figure(1)
hist(means,320:0.25:345)
axis([320 345 0 4000])
figure(2)
hist(means_GPU,320:0.25:345)
axis([320 345 0 4000])

speedup = CPU_time / GPU_time*1000
