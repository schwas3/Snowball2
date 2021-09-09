% Obsolete file, use iframe_detected_threshold_histogram.py instead
clc
clear
close all
%*******************************************************************
% Specify the folder where the files live.
myFolder = '/Users/harrij17/Dropbox/SULI/Snowball/SNOWBALL CROPPED IMAGES/control 03 - 8 bit';
%*******************************************************************
filePattern = fullfile(myFolder, '*.tif');
theFiles = dir(filePattern);
names = split(myFolder,"/");
sizenames = size(names);
%%

% get the size of the files
A = size(theFiles);

% separate the numbers from files, event# and freezingtime#.
for n = 1:A(1)
  baseFileName{n} = theFiles(n).name;
  fullFileName = fullfile(myFolder, baseFileName{n});
  separate = split(baseFileName{n},["_",".tif"]);
  event{n} = separate{1,1};
  time{n} = separate{2,1};
  image = imread(fullFileName);
  imageArray{n} = uint64(image);
%   meaa{n} = mean(imageArray{n},'all');
end
%get how many events are there in the folder
eventID = unique(event);
eventsize = size(eventID);

for m = 1:eventsize(2)
    clear list
    for n = 1 : A(1)
        while ismember(event(n),eventID(m))
            list(n) = n;
            list2{m} = nonzeros(list);   
            listsize{m} = size(list2{m},1);
            B(m) = list2{m}(listsize{m});
            break
        end
    end
end
for m = 1:eventsize(2)
    imgsize{m} = size(imageArray{list2{m}(1)});
end

%calculate the average of first 50 frames of each event as background
for m = 1:eventsize(2)
    X{m} = uint64(zeros(imgsize{m}(1),imgsize{m}(2)));
                %**********************
                for i = (list2{m}(2)) : (list2{m}(2)+49)
                    X{m} = imageArray{i}+X{m};
                end
                XX{m} = X{m}/50;
                %**********************
end   

%subtract the original frame from the background
for m = 1:eventsize(2)
    for n = 1:A(1)
        if eventID{m} == event{n}
            final{n} = imageArray{n} - XX{m};
%             meab{n} = mean(final{n},'all');
%             MAX{n} = max(final{n},[],'all');
        else
            continue
        end
    end
end

%identify the initial frame - 5 x 5
mean = 3;
stdev = 4;%mean and stdev of gaussian curve given in detected_threshold_histogram_auto for run and size
times_thru=0;
to_check=[max(1, mean-stdev), mean, mean+stdev];
for i = 1:3
    Y = zeros(eventsize(2),1);
    for m = 1:eventsize(2)
        for n = list2{m}(2) : list2{m}(listsize{m}-5)
            a=3;
            while a <= imgsize{m}(1)-2
                for b = 3:imgsize{m}(2)-2
                        if (final{n}(a,b)>=to_check(i)) && (final{n}(a+1,b)>=to_check(i)) && (final{n}(a-1,b)>=to_check(i)) && (final{n}(a,b+1)>=to_check(i)) && (final{n}(a,b-1)>=to_check(i)) && (final{n}(a+1,b+1)>=to_check(i)) && (final{n}(a+1,b-1)>=to_check(i)) && (final{n}(a-1,b+1)>=to_check(i)) && (final{n}(a-1,b-1)>=to_check(i)) && (final{n}(a+2,b)>=to_check(i)) && (final{n}(a-2,b)>=to_check(i)) && (final{n}(a,b+2)>=to_check(i)) && (final{n}(a,b-2)>=to_check(i)) && (final{n}(a+2,b+2)>=to_check(i)) && (final{n}(a-2,b+2)>=to_check(i)) && (final{n}(a+2,b-2)>=to_check(i)) && (final{n}(a-2,b-2)>=to_check(i)) && (final{n}(a+1,b+2)>=to_check(i)) && (final{n}(a-1,b+2)>=to_check(i)) && (final{n}(a+1,b-2)>=to_check(i)) && (final{n}(a-1,b-2)>=to_check(i)) && (final{n}(a+2,b+1)>=to_check(i)) && (final{n}(a+2,b-1)>=to_check(i)) && (final{n}(a-2,b+1)>=to_check(i)) && (final{n}(a-2,b-1)>=to_check(i)) && (final{n+1}(a,b)>=to_check(i)) && (final{n+1}(a+1,b)>=to_check(i)) && (final{n+1}(a-1,b)>=to_check(i)) && (final{n+1}(a,b+1)>=to_check(i)) && (final{n+1}(a,b-1)>=to_check(i)) && (final{n+1}(a+1,b+1)>=to_check(i)) && (final{n+1}(a+1,b-1)>=to_check(i)) && (final{n+1}(a-1,b+1)>=to_check(i)) && (final{n+1}(a-1,b-1)>=to_check(i)) && (final{n+1}(a+2,b)>=to_check(i)) && (final{n+1}(a-2,b)>=to_check(i)) && (final{n+1}(a,b+2)>=to_check(i)) && (final{n+1}(a,b-2)>=to_check(i)) && (final{n+1}(a+2,b+2)>=to_check(i)) && (final{n+1}(a-2,b+2)>=to_check(i)) && (final{n+1}(a+2,b-2)>=to_check(i)) && (final{n+1}(a-2,b-2)>=to_check(i)) && (final{n+1}(a+1,b+2)>=to_check(i)) && (final{n+1}(a-1,b+2)>=to_check(i)) && (final{n+1}(a+1,b-2)>=to_check(i)) && (final{n+1}(a-1,b-2)>=to_check(i)) && (final{n+1}(a+2,b+1)>=to_check(i)) && (final{n+1}(a+2,b-1)>=to_check(i)) && (final{n+1}(a-2,b+1)>=to_check(i)) && (final{n+1}(a-2,b-1)>=to_check(i)) && (final{n+2}(a,b)>=to_check(i)) && (final{n+2}(a+1,b)>=to_check(i)) && (final{n+2}(a-1,b)>=to_check(i)) && (final{n+2}(a,b+1)>=to_check(i)) && (final{n+2}(a,b-1)>=to_check(i)) && (final{n+2}(a+1,b+1)>=to_check(i)) && (final{n+2}(a+1,b-1)>=to_check(i)) && (final{n+2}(a-1,b+1)>=to_check(i)) && (final{n+2}(a-1,b-1)>=to_check(i)) && (final{n+2}(a+2,b)>=to_check(i)) && (final{n+2}(a-2,b)>=to_check(i)) && (final{n+2}(a,b+2)>=to_check(i)) && (final{n+2}(a,b-2)>=to_check(i)) && (final{n+2}(a+2,b+2)>=to_check(i)) && (final{n+2}(a-2,b+2)>=to_check(i)) && (final{n+2}(a+2,b-2)>=to_check(i)) && (final{n+2}(a-2,b-2)>=to_check(i)) && (final{n+2}(a+1,b+2)>=to_check(i)) && (final{n+2}(a-1,b+2)>=to_check(i)) && (final{n+2}(a+1,b-2)>=to_check(i)) && (final{n+2}(a-1,b-2)>=to_check(i)) && (final{n+2}(a+2,b+1)>=to_check(i)) && (final{n+2}(a+2,b-1)>=to_check(i)) && (final{n+2}(a-2,b+1)>=to_check(i)) && (final{n+2}(a-2,b-1)>=to_check(i)) && (final{n+3}(a,b)>=to_check(i)) && (final{n+3}(a+1,b)>=to_check(i)) && (final{n+3}(a-1,b)>=to_check(i)) && (final{n+3}(a,b+1)>=to_check(i)) && (final{n+3}(a,b-1)>=to_check(i)) && (final{n+3}(a+1,b+1)>=to_check(i)) && (final{n+3}(a+1,b-1)>=to_check(i)) && (final{n+3}(a-1,b+1)>=to_check(i)) && (final{n+3}(a-1,b-1)>=to_check(i)) && (final{n+3}(a+2,b)>=to_check(i)) && (final{n+3}(a-2,b)>=to_check(i)) && (final{n+3}(a,b+2)>=to_check(i)) && (final{n+3}(a,b-2)>=to_check(i)) && (final{n+3}(a+2,b+2)>=to_check(i)) && (final{n+3}(a-2,b+2)>=to_check(i)) && (final{n+3}(a+2,b-2)>=to_check(i)) && (final{n+3}(a-2,b-2)>=to_check(i)) && (final{n+3}(a+1,b+2)>=to_check(i)) && (final{n+3}(a-1,b+2)>=to_check(i)) && (final{n+3}(a+1,b-2)>=to_check(i)) && (final{n+3}(a-1,b-2)>=to_check(i)) && (final{n+3}(a+2,b+1)>=to_check(i)) && (final{n+3}(a+2,b-1)>=to_check(i)) && (final{n+3}(a-2,b+1)>=to_check(i)) && (final{n+3}(a-2,b-1)>=to_check(i)) && (final{n+4}(a,b)>=to_check(i)) && (final{n+4}(a+1,b)>=to_check(i)) && (final{n+4}(a-1,b)>=to_check(i)) && (final{n+4}(a,b+1)>=to_check(i)) && (final{n+4}(a,b-1)>=to_check(i)) && (final{n+4}(a+1,b+1)>=to_check(i)) && (final{n+4}(a+1,b-1)>=to_check(i)) && (final{n+4}(a-1,b+1)>=to_check(i)) && (final{n+4}(a-1,b-1)>=to_check(i)) && (final{n+4}(a+2,b)>=to_check(i)) && (final{n+4}(a-2,b)>=to_check(i)) && (final{n+4}(a,b+2)>=to_check(i)) && (final{n+4}(a,b-2)>=to_check(i)) && (final{n+4}(a+2,b+2)>=to_check(i)) && (final{n+4}(a-2,b+2)>=to_check(i)) && (final{n+4}(a+2,b-2)>=to_check(i)) && (final{n+4}(a-2,b-2)>=to_check(i)) && (final{n+4}(a+1,b+2)>=to_check(i)) && (final{n+4}(a-1,b+2)>=to_check(i)) && (final{n+4}(a+1,b-2)>=to_check(i)) && (final{n+4}(a-1,b-2)>=to_check(i)) && (final{n+4}(a+2,b+1)>=to_check(i)) && (final{n+4}(a+2,b-1)>=to_check(i)) && (final{n+4}(a-2,b+1)>=to_check(i)) && (final{n+4}(a-2,b-1)>=to_check(i)) && (final{n+5}(a,b)>=to_check(i)) && (final{n+5}(a+1,b)>=to_check(i)) && (final{n+5}(a-1,b)>=to_check(i)) && (final{n+5}(a,b+1)>=to_check(i)) && (final{n+5}(a,b-1)>=to_check(i)) && (final{n+5}(a+1,b+1)>=to_check(i)) && (final{n+5}(a+1,b-1)>=to_check(i)) && (final{n+5}(a-1,b+1)>=to_check(i)) && (final{n+5}(a-1,b-1)>=to_check(i)) && (final{n+5}(a+2,b)>=to_check(i)) && (final{n+5}(a-2,b)>=to_check(i)) && (final{n+5}(a,b+2)>=to_check(i)) && (final{n+5}(a,b-2)>=to_check(i)) && (final{n+5}(a+2,b+2)>=to_check(i)) && (final{n+5}(a-2,b+2)>=to_check(i)) && (final{n+5}(a+2,b-2)>=to_check(i)) && (final{n+5}(a-2,b-2)>=to_check(i)) && (final{n+5}(a+1,b+2)>=to_check(i)) && (final{n+5}(a-1,b+2)>=to_check(i)) && (final{n+5}(a+1,b-2)>=to_check(i)) && (final{n+5}(a-1,b-2)>=to_check(i)) && (final{n+5}(a+2,b+1)>=to_check(i)) && (final{n+5}(a+2,b-1)>=to_check(i)) && (final{n+5}(a-2,b+1)>=to_check(i)) && (final{n+5}(a-2,b-1)>=to_check(i))
                         Y(m) = n;
                         break
                        end
                end

                if Y(m) == 0
                    a=a+1;
                else
                    break
                end
            end
            if (Y(m) == 0) && (n == list2{m}(listsize{m}-5))
                Y(m) = B(m);
                break
            elseif Y(m) ~=0
                break
            end
        end
    end
    times_thru = times_thru+1;
    if times_thru == 1
        comb = [names{sizenames},'_frame_threshold_5x5_lower.txt'];
    elseif times_thru == 2
        comb = [names{sizenames},'_frame_threshold_5x5_mean.txt'];
    else
        comb = [names{sizenames},'_frame_threshold_5x5_upper.txt'];
    end
    resultfile = join(comb);
    writematrix(Y, resultfile, 'Delimiter','tab')
    save(names{sizenames})
end

%identify the initial frame - 3 x 3
mean = 5;
stdev = 5;%mean and stdev of gaussian curve given in detected_threshold_histogram_auto for run and size
times_thru=0;
to_check=[max(1, mean-stdev), mean, mean+stdev];
for i = 1:3
    Y = zeros(eventsize(2),1);
    Z = zeros(eventsize(2),1);
    for m = 1:eventsize(2)
        for n = list2{m}(2) : list2{m}(listsize{m}-5)
            a=2;
            while a <= imgsize{m}(1)-1
                for b = 2:imgsize{m}(2)-1
    %                     if (final{n}(a,b)>=to_check(i)) && (final{n+1}(a,b)>=to_check(i)) && (final{n+2}(a,b)>=to_check(i)) && (final{n+3}(a,b)>=to_check(i)) && (final{n+4}(a,b)>=to_check(i)) %&& (final{n+5}(a,b)>=to_check(i))
                       if(final{n}(a,b)>=to_check(i)) && (final{n}(a+1,b)>=to_check(i)) && (final{n}(a-1,b)>=to_check(i)) && (final{n}(a,b+1)>=to_check(i)) && (final{n}(a,b-1)>=to_check(i)) && (final{n}(a+1,b+1)>=to_check(i)) && (final{n}(a+1,b-1)>=to_check(i)) && (final{n}(a-1,b+1)>=to_check(i)) && (final{n}(a-1,b-1)>=to_check(i)) &&(final{n+1}(a,b)>=to_check(i)) && (final{n+1}(a+1,b)>=to_check(i)) && (final{n+1}(a-1,b)>=to_check(i)) && (final{n+1}(a,b+1)>=to_check(i)) && (final{n+1}(a,b-1)>=to_check(i)) && (final{n+1}(a+1,b+1)>=to_check(i)) && (final{n+1}(a+1,b-1)>=to_check(i)) && (final{n+1}(a-1,b+1)>=to_check(i)) && (final{n+1}(a-1,b-1)>=to_check(i)) &&(final{n+2}(a,b)>=to_check(i)) && (final{n+2}(a+1,b)>=to_check(i)) && (final{n+2}(a-1,b)>=to_check(i)) && (final{n+2}(a,b+1)>=to_check(i)) && (final{n+2}(a,b-1)>=to_check(i)) && (final{n+2}(a+1,b+1)>=to_check(i)) && (final{n+2}(a+1,b-1)>=to_check(i)) && (final{n+2}(a-1,b+1)>=to_check(i)) && (final{n+2}(a-1,b-1)>=to_check(i)) &&(final{n+3}(a,b)>=to_check(i)) && (final{n+3}(a+1,b)>=to_check(i)) && (final{n+3}(a-1,b)>=to_check(i)) && (final{n+3}(a,b+1)>=to_check(i)) && (final{n+3}(a,b-1)>=to_check(i)) && (final{n+3}(a+1,b+1)>=to_check(i)) && (final{n+3}(a+1,b-1)>=to_check(i)) && (final{n+3}(a-1,b+1)>=to_check(i)) && (final{n+3}(a-1,b-1)>=to_check(i)) &&(final{n+4}(a,b)>=to_check(i)) && (final{n+4}(a+1,b)>=to_check(i)) && (final{n+4}(a-1,b)>=to_check(i)) && (final{n+4}(a,b+1)>=to_check(i)) && (final{n+4}(a,b-1)>=to_check(i)) && (final{n+4}(a+1,b+1)>=to_check(i)) && (final{n+4}(a+1,b-1)>=to_check(i)) && (final{n+4}(a-1,b+1)>=to_check(i)) && (final{n+4}(a-1,b-1)>=to_check(i)) &&(final{n+5}(a,b)>=to_check(i)) && (final{n+5}(a+1,b)>=to_check(i)) && (final{n+5}(a-1,b)>=to_check(i)) && (final{n+5}(a,b+1)>=to_check(i)) && (final{n+5}(a,b-1)>=to_check(i)) && (final{n+5}(a+1,b+1)>=to_check(i)) && (final{n+5}(a+1,b-1)>=to_check(i)) && (final{n+5}(a-1,b+1)>=to_check(i)) && (final{n+5}(a-1,b-1)>=to_check(i))                          
                        Y(m) = n;
                        break
                       end
                end

                if Y(m) == 0
                    a=a+1;
                else
                    break
                end
            end
            if (Y(m) == 0) && (n == list2{m}(listsize{m}-5))
                Y(m) = B(m);
                break
            elseif Y(m) ~=0
                break
            end
        end
    end
    times_thru = times_thru+1;
    if times_thru == 1
        comb = [names{sizenames},'_frame_threshold_3x3_lower.txt'];
    elseif times_thru == 2
        comb = [names{sizenames},'_frame_threshold_3x3_mean.txt'];
    else
        comb = [names{sizenames},'_frame_threshold_3x3_upper.txt'];
    end
    resultfile = join(comb);
    writematrix(Y, resultfile, 'Delimiter','tab')
    save(names{sizenames})
end

%identify the initial frame - 2 x 2
mean = 9;
stdev = 8;%mean and stdev of gaussian curve given in detected_threshold_histogram_auto for run and size
times_thru=0;
to_check=[max(1, mean-stdev), mean, mean+stdev];
for i = 1:3
    Y = zeros(eventsize(2),1);
    Z = zeros(eventsize(2),1);
    for m = 1:eventsize(2)
        for n = list2{m}(2) : list2{m}(listsize{m}-5)
            if i ~= 1 && Y(m)==B(m)
                Y(m)=B(m);
                break
            end

            a=1;
            while a <= imgsize{m}(1)-1
                for b = 1:imgsize{m}(2)-1
                       if(final{n}(a,b)>=to_check(i)) && (final{n}(a+1,b)>=to_check(i)) && (final{n}(a,b+1)>=to_check(i)) && (final{n}(a+1,b+1)>=to_check(i)) && (final{n+1}(a,b)>=to_check(i)) && (final{n+1}(a+1,b)>=to_check(i)) && (final{n+1}(a,b+1)>=to_check(i)) && (final{n+1}(a+1,b+1)>=to_check(i)) && (final{n+2}(a,b)>=to_check(i)) && (final{n+2}(a+1,b)>=to_check(i)) && (final{n+2}(a,b+1)>=to_check(i)) && (final{n+2}(a+1,b+1)>=to_check(i)) && (final{n+3}(a,b)>=to_check(i)) && (final{n+3}(a+1,b)>=to_check(i)) && (final{n+3}(a,b+1)>=to_check(i)) && (final{n+3}(a+1,b+1)>=to_check(i)) && (final{n+4}(a,b)>=to_check(i)) && (final{n+4}(a+1,b)>=to_check(i)) && (final{n+4}(a,b+1)>=to_check(i)) && (final{n+4}(a+1,b+1)>=to_check(i)) && (final{n+5}(a,b)>=to_check(i)) && (final{n+5}(a+1,b)>=to_check(i)) && (final{n+5}(a,b+1)>=to_check(i)) && (final{n+5}(a+1,b+1)>=to_check(i))                          
                        Y(m) = n;
                        break
                       end
                end

                if Y(m) == 0
                    a=a+1;
                else
                    break
                end
            end
            if (Y(m) == 0) && (n == list2{m}(listsize{m}-5))
                Y(m) = B(m);
                break
            elseif Y(m) ~=0
                break
            end
        end
    end
    times_thru = times_thru+1;
    if times_thru == 1
        comb = [names{sizenames},'_frame_threshold_2x2_lower.txt'];
    elseif times_thru == 2
        comb = [names{sizenames},'_frame_threshold_2x2_mean.txt'];
    else
        comb = [names{sizenames},'_frame_threshold_2x2_upper.txt'];
    end
    resultfile = join(comb);
    writematrix(Y, resultfile, 'Delimiter','tab')
    save(names{sizenames})
end
