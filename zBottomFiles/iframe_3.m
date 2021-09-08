% Developed by Yujia Huang, MSc in progress, UAlbany. Prototype of iframe.m
% (use that instead)
clc
clear
close all
%*******************************************************************
% Specify the folder where the files live.
myFolder = '/Users/harrij17/Dropbox/SULI/Snowball/SNOWBALL CROPPED IMAGES/control 02 - 8 bit';
%*******************************************************************
filePattern = fullfile(myFolder, '*.tif');
theFiles = dir(filePattern);
names = split(myFolder,"/");
sizenames = size(names);

comb = [names{sizenames},'_3_frame.txt'];
comblocation = [comb,'_location.csv'];
resultfile = join(comb);

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
imgsize = size(imageArray{1});
for m = 1:eventsize(2)
    clear list
    for n = 1 : A(1)-1
        while ismember(event(n),eventID(m))
            list(n) = n;
            list2{m} = nonzeros(list);   
            listsize{m} = size(list2{m},1);
            B(m) = list2{m}(listsize{m});
            break
        end
    end
end

%calculate the average of first 50 frames of each event as background
for m = 1:eventsize(2)
    X{m} = uint64(zeros(imgsize(1),imgsize(2)));
                %**********************
                for i = (list2{m}(2)) : (list2{m}(2)+49 )
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

%identify the initial frame
Y = num2cell(zeros(eventsize(2),255));
for i = 1:255
    for m = [1 floor(eventsize(2)/2) eventsize(2)]
        for n = list2{m}(2) : list2{m}(listsize{m})
            a=2;
            while a <= imgsize(1)-1
                for b = 2:imgsize(2)-1
%                     if (final{n}(a,b)>=i) && (final{n+1}(a,b)>=i) && (final{n+2}(a,b)>=i) && (final{n+3}(a,b)>=i) && (final{n+4}(a,b)>=i) %&& (final{n+5}(a,b)>=i)
%identify the nucleation site with 6 frames in series with adjacent pixel
                        if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) && (final{n}(a-1,b-1)>=i) &&(final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) && (final{n+1}(a-1,b-1)>=i) &&(final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) && (final{n+2}(a-1,b-1)>=i) &&(final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) && (final{n+3}(a-1,b-1)>=i) &&(final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) && (final{n+4}(a-1,b-1)>=i) &&(final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) && (final{n+5}(a-1,b-1)>=i)                          
                        Y{m,i} = n;
                        Yp{m,i}=[a,b];
                        break
                    end
                end
                
%if there are no such nucleation site found in this line, move to the next line
                if Y{m,i} == 0
                    a=a+1;
                else
                    break
                end
            end
            %if there are no such nucleation site in the entire event, set
            %the nucleation site at the end of the event
            if (Y{m,i} == 0) && (n == list2{m}(listsize{m}))
                Y{m,i} = B(m);
                Yp{m,i}=[0,0];
                break
            elseif Y{m,i} ~=0
                break
            end
        end
    end
end

fileID = fopen('/Users/harrij17/Dropbox/SULI/Snowball/SNOWBALL CROPPED IMAGES/control 02 - 8 bit/knowncontrol02_3.txt','r');
C = fscanf(fileID,'%f');
fclose(fileID);
Z = zeros(eventsize(2),255);

for m = [1 floor(eventsize(2)/2) eventsize(2)]
    for i = 1:255
            Z(m,i)  = Y{m,i} - C(m);
    end
end
average = mean(Z);
uncertainty = std(Z)/sqrt(eventsize(2));

for i = 1:255
        result{i,:,:,:} = [i, average(i),0 , uncertainty(i)];
end
writecell(result, resultfile,'Delimiter','tab')
%writecell(Yp, comblocation)
save(names{sizenames})