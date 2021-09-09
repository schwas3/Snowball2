% Originally developed by Yujia Huang, MSc in progress, UAlbany. Main
% analysis file. Change folder and answer key location (myfolder, fileID),
% detection area to use (comb, init frame identification) before running
% code. Returns three cells of dimensions of data sets*255 containing frames
% detected, their deviation from the answer key, and coordinates of
% snowball. Also returns 255*1 cell (main result file) containing threshold
% used, average deviation, uncertainty (always 0) and standard deviation of
% the deviation from the answer key.
clc
clear
close all
%*******************************************************************
% Specify the folder where the files live.
myFolder = '/Users/harrij17/Dropbox/SULI/Snowball/SNOWBALL CROPPED IMAGES/fiesta front w Be 10 - 8 bit';
%*******************************************************************
% Reads images into file, sets up names of result files.
filePattern = fullfile(myFolder, '*.tif');
theFiles = dir(filePattern);
names = split(myFolder, "/");
sizenames = size(names);
comb = [names{sizenames}, '_frame_8x8.txt'];
comblocation = [comb, '_location.csv'];
combframes = [comb, '_detected_frames.txt'];
combframesZ = [comb, '_subtracted_frames.txt'];
resultfile = join(comb);
%%
% get the size of the files
A = size(theFiles);

% separate the numbers from files, dataset# and freezingtime#. Also reads
% them into pixel values of 0 (black) to 255 (white)
for n = 1:A(1)
    baseFileName{n} = theFiles(n).name;
    fullFileName = fullfile(myFolder, baseFileName{n});
    separate = split(baseFileName{n}, ["_", ".tif"]);
    event{n} = separate{1, 1};
    time{n} = separate{2, 1};
    image = imread(fullFileName);
    imageArray{n} = uint64(image);
    %   meaa{n} = mean(imageArray{n},'all');
end

%get how many data sets are there in the folder
eventID = unique(event);
eventsize = size(eventID);
%sort images into data sets, using the IDs as delimiters
for m = 1:eventsize(2)
    clear list

    for n = 1:A(1)

        while ismember(event(n), eventID(m))
            list(n) = n;
            list2{m} = nonzeros(list);
            listsize{m} = size(list2{m}, 1);
            B(m) = list2{m}(listsize{m});
            break
        end

    end

end

for m = 1:eventsize(2)
    imgsize{m} = size(imageArray{list2{m}(1)});
end

%calculate the average of first 50 frames of each data set as background
for m = 1:eventsize(2)
    X{m} = uint64(zeros(imgsize{m}(1), imgsize{m}(2)));
    %**********************
    for i = (list2{m}(2)):(list2{m}(2) + 49)
        X{m} = imageArray{i} + X{m};
    end

    XX{m} = X{m} / 50;
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

%%
%identify the initial frame
Y = num2cell(zeros(eventsize(2), 255));
%chooses a pixel value from 1 to 255 (threshold), checks the images in each data
%set for it
for i = 1:255

    for m = 1:eventsize(2)

        for n = list2{m}(2):list2{m}(listsize{m} - 5)
            % skips checking if no snowball was found at previous threshold,
            % assigns default values of largest image # in data set for
            % Y position, [0, 0] for Yp position
            if i ~= 1 && Y{m, i - 1} == B(m)
                Y{m, i} = B(m);
                Yp{m, i} = [0, 0];
                break
            end

            %detection area starts at top left, moves across the row, then
            %down a row if no snowball found, repeat

            %tip: use ctrl+r to comment out sections of code

            %if all pixels in the area have pixel values equal to or above the
            %threshold for six frames straight, snowball detected - frame
            %recorded in corresponding Y position, snowball position
            %(top left of innermost 2x2 area for even side length,
            %central pixel for odd) recorded in [row, column] form in
            %corresponding Yp position

            %%for 14 x 14
            %              a=7;
            %              while a <= imgsize{m}(1)-7
            %                  to_check_a=[a-6, a-5, a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4, a+5, a+6, a+7];
            %                  for b = 7:imgsize{m}(2)-7
            %                      detected="Y";
            %                      for c = 1:14
            %                          if (final{n}(to_check_a(c),b-6)<i) || (final{n}(to_check_a(c),b-5)<i) || (final{n}(to_check_a(c),b-4)<i) || (final{n}(to_check_a(c),b-3)<i) || (final{n}(to_check_a(c),b-2)<i) || (final{n}(to_check_a(c),b-1)<i) || (final{n}(to_check_a(c),b)<i) || (final{n}(to_check_a(c),b+1)<i) || (final{n}(to_check_a(c),b+2)<i) || (final{n}(to_check_a(c),b+3)<i) || (final{n}(to_check_a(c),b+4)<i) || (final{n}(to_check_a(c),b+5)<i) || (final{n}(to_check_a(c),b+6)<i) || (final{n}(to_check_a(c),b+7)<i) || (final{n+1}(to_check_a(c),b-6)<i) || (final{n+1}(to_check_a(c),b-5)<i) || (final{n+1}(to_check_a(c),b-4)<i) || (final{n+1}(to_check_a(c),b-3)<i) || (final{n+1}(to_check_a(c),b-2)<i) || (final{n+1}(to_check_a(c),b-1)<i) || (final{n+1}(to_check_a(c),b)<i) || (final{n+1}(to_check_a(c),b+1)<i) || (final{n+1}(to_check_a(c),b+2)<i) || (final{n+1}(to_check_a(c),b+3)<i) || (final{n+1}(to_check_a(c),b+4)<i) || (final{n+1}(to_check_a(c),b+5)<i) || (final{n+1}(to_check_a(c),b+6)<i) || (final{n+1}(to_check_a(c),b+7)<i) || (final{n+2}(to_check_a(c),b-6)<i) || (final{n+2}(to_check_a(c),b-5)<i) || (final{n+2}(to_check_a(c),b-4)<i) || (final{n+2}(to_check_a(c),b-3)<i) || (final{n+2}(to_check_a(c),b-2)<i) || (final{n+2}(to_check_a(c),b-1)<i) || (final{n+2}(to_check_a(c),b)<i) || (final{n+2}(to_check_a(c),b+1)<i) || (final{n+2}(to_check_a(c),b+2)<i) || (final{n+2}(to_check_a(c),b+3)<i) || (final{n+2}(to_check_a(c),b+4)<i) || (final{n+2}(to_check_a(c),b+5)<i) || (final{n+2}(to_check_a(c),b+6)<i) || (final{n+2}(to_check_a(c),b+7)<i) || (final{n+3}(to_check_a(c),b-6)<i) || (final{n+3}(to_check_a(c),b-5)<i) || (final{n+3}(to_check_a(c),b-4)<i) || (final{n+3}(to_check_a(c),b-3)<i) || (final{n+3}(to_check_a(c),b-2)<i) || (final{n+3}(to_check_a(c),b-1)<i) || (final{n+3}(to_check_a(c),b)<i) || (final{n+3}(to_check_a(c),b+1)<i) || (final{n+3}(to_check_a(c),b+2)<i) || (final{n+3}(to_check_a(c),b+3)<i) || (final{n+3}(to_check_a(c),b+4)<i) || (final{n+3}(to_check_a(c),b+5)<i) || (final{n+3}(to_check_a(c),b+6)<i) || (final{n+3}(to_check_a(c),b+7)<i) || (final{n+4}(to_check_a(c),b-6)<i) || (final{n+4}(to_check_a(c),b-5)<i) || (final{n+4}(to_check_a(c),b-4)<i) || (final{n+4}(to_check_a(c),b-3)<i) || (final{n+4}(to_check_a(c),b-2)<i) || (final{n+4}(to_check_a(c),b-1)<i) || (final{n+4}(to_check_a(c),b)<i) || (final{n+4}(to_check_a(c),b+1)<i) || (final{n+4}(to_check_a(c),b+2)<i) || (final{n+4}(to_check_a(c),b+3)<i) || (final{n+4}(to_check_a(c),b+4)<i) || (final{n+4}(to_check_a(c),b+5)<i) || (final{n+4}(to_check_a(c),b+6)<i) || (final{n+4}(to_check_a(c),b+7)<i) || (final{n+5}(to_check_a(c),b-6)<i) || (final{n+5}(to_check_a(c),b-5)<i) || (final{n+5}(to_check_a(c),b-4)<i) || (final{n+5}(to_check_a(c),b-3)<i) || (final{n+5}(to_check_a(c),b-2)<i) || (final{n+5}(to_check_a(c),b-1)<i) || (final{n+5}(to_check_a(c),b)<i) || (final{n+5}(to_check_a(c),b+1)<i) || (final{n+5}(to_check_a(c),b+2)<i) || (final{n+5}(to_check_a(c),b+3)<i) || (final{n+5}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i) || (final{n+5}(to_check_a(c),b+6)<i) || (final{n+5}(to_check_a(c),b+7)<i)
            %                              detected="N";
            %                              break
            %                          end
            %                      end
            %                      if detected=="Y"
            %                          Y{m,i} = n;
            %                          Yp{m,i}=[a,b];
            %                          break
            %                      end
            %                  end

            %%for 13 x 13
            %             a=7;
            %             while a <= imgsize{m}(1)-6
            %                 to_check_a=[a-6, a-5, a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4, a+5, a+6];
            %                 for b = 7:imgsize{m}(2)-6
            %                     detected="Y";
            %                     for c = 1:13
            %                         if (final{n}(to_check_a(c),b-6)<i) || (final{n}(to_check_a(c),b-5)<i) || (final{n}(to_check_a(c),b-4)<i) || (final{n}(to_check_a(c),b-3)<i) || (final{n}(to_check_a(c),b-2)<i) || (final{n}(to_check_a(c),b-1)<i) || (final{n}(to_check_a(c),b)<i) || (final{n}(to_check_a(c),b+1)<i) || (final{n}(to_check_a(c),b+2)<i) || (final{n}(to_check_a(c),b+3)<i) || (final{n}(to_check_a(c),b+4)<i) || (final{n}(to_check_a(c),b+5)<i) || (final{n}(to_check_a(c),b+6)<i) || (final{n+1}(to_check_a(c),b-6)<i) || (final{n+1}(to_check_a(c),b-5)<i) || (final{n+1}(to_check_a(c),b-4)<i) || (final{n+1}(to_check_a(c),b-3)<i) || (final{n+1}(to_check_a(c),b-2)<i) || (final{n+1}(to_check_a(c),b-1)<i) || (final{n+1}(to_check_a(c),b)<i) || (final{n+1}(to_check_a(c),b+1)<i) || (final{n+1}(to_check_a(c),b+2)<i) || (final{n+1}(to_check_a(c),b+3)<i) || (final{n+1}(to_check_a(c),b+4)<i) || (final{n+1}(to_check_a(c),b+5)<i) || (final{n+1}(to_check_a(c),b+6)<i) || (final{n+2}(to_check_a(c),b-6)<i) || (final{n+2}(to_check_a(c),b-5)<i) || (final{n+2}(to_check_a(c),b-4)<i) || (final{n+2}(to_check_a(c),b-3)<i) || (final{n+2}(to_check_a(c),b-2)<i) || (final{n+2}(to_check_a(c),b-1)<i) || (final{n+2}(to_check_a(c),b)<i) || (final{n+2}(to_check_a(c),b+1)<i) || (final{n+2}(to_check_a(c),b+2)<i) || (final{n+2}(to_check_a(c),b+3)<i) || (final{n+2}(to_check_a(c),b+4)<i) || (final{n+2}(to_check_a(c),b+5)<i) || (final{n+2}(to_check_a(c),b+6)<i) || (final{n+3}(to_check_a(c),b-6)<i) || (final{n+3}(to_check_a(c),b-5)<i) || (final{n+3}(to_check_a(c),b-4)<i) || (final{n+3}(to_check_a(c),b-3)<i) || (final{n+3}(to_check_a(c),b-2)<i) || (final{n+3}(to_check_a(c),b-1)<i) || (final{n+3}(to_check_a(c),b)<i) || (final{n+3}(to_check_a(c),b+1)<i) || (final{n+3}(to_check_a(c),b+2)<i) || (final{n+3}(to_check_a(c),b+3)<i) || (final{n+3}(to_check_a(c),b+4)<i) || (final{n+3}(to_check_a(c),b+5)<i) || (final{n+3}(to_check_a(c),b+6)<i) || (final{n+4}(to_check_a(c),b-6)<i) || (final{n+4}(to_check_a(c),b-5)<i) || (final{n+4}(to_check_a(c),b-4)<i) || (final{n+4}(to_check_a(c),b-3)<i) || (final{n+4}(to_check_a(c),b-2)<i) || (final{n+4}(to_check_a(c),b-1)<i) || (final{n+4}(to_check_a(c),b)<i) || (final{n+4}(to_check_a(c),b+1)<i) || (final{n+4}(to_check_a(c),b+2)<i) || (final{n+4}(to_check_a(c),b+3)<i) || (final{n+4}(to_check_a(c),b+4)<i) || (final{n+4}(to_check_a(c),b+5)<i) || (final{n+4}(to_check_a(c),b+6)<i) || (final{n+5}(to_check_a(c),b-6)<i) || (final{n+5}(to_check_a(c),b-5)<i) || (final{n+5}(to_check_a(c),b-4)<i) || (final{n+5}(to_check_a(c),b-3)<i) || (final{n+5}(to_check_a(c),b-2)<i) || (final{n+5}(to_check_a(c),b-1)<i) || (final{n+5}(to_check_a(c),b)<i) || (final{n+5}(to_check_a(c),b+1)<i) || (final{n+5}(to_check_a(c),b+2)<i) || (final{n+5}(to_check_a(c),b+3)<i) || (final{n+5}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i) || (final{n+5}(to_check_a(c),b+6)<i)
            %                             detected="N";
            %                             break
            %                         end
            %                     end
            %                     if detected=="Y"
            %                         Y{m,i} = n;
            %                         Yp{m,i}=[a,b];
            %                         break
            %                     end
            %                 end

            %%for 12 x 12
            %             a=6;
            %             while a <= imgsize{m}(1)-6
            %                    to_check_a=[a-5, a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4, a+5, a+6];
            %                    for b = 6:imgsize{m}(2)-6
            %                        detected="Y";
            %                        for c = 1:12
            %                            if (final{n}(to_check_a(c),b-5)<i) || (final{n}(to_check_a(c),b-4)<i) || (final{n}(to_check_a(c),b-3)<i) || (final{n}(to_check_a(c),b-2)<i) || (final{n}(to_check_a(c),b-1)<i) || (final{n}(to_check_a(c),b)<i) || (final{n}(to_check_a(c),b+1)<i) || (final{n}(to_check_a(c),b+2)<i) || (final{n}(to_check_a(c),b+3)<i) || (final{n}(to_check_a(c),b+4)<i) || (final{n}(to_check_a(c),b+5)<i) || (final{n}(to_check_a(c),b+6)<i) || (final{n+1}(to_check_a(c),b-5)<i) || (final{n+1}(to_check_a(c),b-4)<i) || (final{n+1}(to_check_a(c),b-3)<i) || (final{n+1}(to_check_a(c),b-2)<i) || (final{n+1}(to_check_a(c),b-1)<i) || (final{n+1}(to_check_a(c),b)<i) || (final{n+1}(to_check_a(c),b+1)<i) || (final{n+1}(to_check_a(c),b+2)<i) || (final{n+1}(to_check_a(c),b+3)<i) || (final{n+1}(to_check_a(c),b+4)<i) || (final{n+1}(to_check_a(c),b+5)<i) || (final{n+1}(to_check_a(c),b+6)<i) || (final{n+2}(to_check_a(c),b-5)<i) || (final{n+2}(to_check_a(c),b-4)<i) || (final{n+2}(to_check_a(c),b-3)<i) || (final{n+2}(to_check_a(c),b-2)<i) || (final{n+2}(to_check_a(c),b-1)<i) || (final{n+2}(to_check_a(c),b)<i) || (final{n+2}(to_check_a(c),b+1)<i) || (final{n+2}(to_check_a(c),b+2)<i) || (final{n+2}(to_check_a(c),b+3)<i) || (final{n+2}(to_check_a(c),b+4)<i) || (final{n+2}(to_check_a(c),b+5)<i) || (final{n+2}(to_check_a(c),b+6)<i) || (final{n+3}(to_check_a(c),b-5)<i) || (final{n+3}(to_check_a(c),b-4)<i) || (final{n+3}(to_check_a(c),b-3)<i) || (final{n+3}(to_check_a(c),b-2)<i) || (final{n+3}(to_check_a(c),b-1)<i) || (final{n+3}(to_check_a(c),b)<i) || (final{n+3}(to_check_a(c),b+1)<i) || (final{n+3}(to_check_a(c),b+2)<i) || (final{n+3}(to_check_a(c),b+3)<i) || (final{n+3}(to_check_a(c),b+4)<i) || (final{n+3}(to_check_a(c),b+5)<i) || (final{n+3}(to_check_a(c),b+6)<i) || (final{n+4}(to_check_a(c),b-5)<i) || (final{n+4}(to_check_a(c),b-4)<i) || (final{n+4}(to_check_a(c),b-3)<i) || (final{n+4}(to_check_a(c),b-2)<i) || (final{n+4}(to_check_a(c),b-1)<i) || (final{n+4}(to_check_a(c),b)<i) || (final{n+4}(to_check_a(c),b+1)<i) || (final{n+4}(to_check_a(c),b+2)<i) || (final{n+4}(to_check_a(c),b+3)<i) || (final{n+4}(to_check_a(c),b+4)<i) || (final{n+4}(to_check_a(c),b+5)<i) || (final{n+4}(to_check_a(c),b+6)<i) || (final{n+5}(to_check_a(c),b-5)<i) || (final{n+5}(to_check_a(c),b-4)<i) || (final{n+5}(to_check_a(c),b-3)<i) || (final{n+5}(to_check_a(c),b-2)<i) || (final{n+5}(to_check_a(c),b-1)<i) || (final{n+5}(to_check_a(c),b)<i) || (final{n+5}(to_check_a(c),b+1)<i) || (final{n+5}(to_check_a(c),b+2)<i) || (final{n+5}(to_check_a(c),b+3)<i) || (final{n+5}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i) || (final{n+5}(to_check_a(c),b+6)<i)
            %                                detected="N";
            %                                break
            %                            end
            %                        end
            %                        if detected=="Y"
            %                            Y{m,i} = n;
            %                            Yp{m,i}=[a,b];
            %                            break
            %                        end
            %                    end

            %%for 11 x 11
            %             a=6;
            %             while a <= imgsize{m}(1)-5
            %                 to_check_a=[a-5, a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4, a+5];
            %                 for b = 6:imgsize{m}(2)-5
            %                     detected="Y";
            %                     for c = 1:11
            %                         if (final{n}(to_check_a(c),b-5)<i) || (final{n}(to_check_a(c),b-4)<i) || (final{n}(to_check_a(c),b-3)<i) || (final{n}(to_check_a(c),b-2)<i) || (final{n}(to_check_a(c),b-1)<i) || (final{n}(to_check_a(c),b)<i) || (final{n}(to_check_a(c),b+1)<i) || (final{n}(to_check_a(c),b+2)<i) || (final{n}(to_check_a(c),b+3)<i) || (final{n}(to_check_a(c),b+4)<i) || (final{n}(to_check_a(c),b+5)<i) || (final{n+1}(to_check_a(c),b-5)<i) || (final{n+1}(to_check_a(c),b-4)<i) || (final{n+1}(to_check_a(c),b-3)<i) || (final{n+1}(to_check_a(c),b-2)<i) || (final{n+1}(to_check_a(c),b-1)<i) || (final{n+1}(to_check_a(c),b)<i) || (final{n+1}(to_check_a(c),b+1)<i) || (final{n+1}(to_check_a(c),b+2)<i) || (final{n+1}(to_check_a(c),b+3)<i) || (final{n+1}(to_check_a(c),b+4)<i) || (final{n+1}(to_check_a(c),b+5)<i) || (final{n+2}(to_check_a(c),b-5)<i) || (final{n+2}(to_check_a(c),b-4)<i) || (final{n+2}(to_check_a(c),b-3)<i) || (final{n+2}(to_check_a(c),b-2)<i) || (final{n+2}(to_check_a(c),b-1)<i) || (final{n+2}(to_check_a(c),b)<i) || (final{n+2}(to_check_a(c),b+1)<i) || (final{n+2}(to_check_a(c),b+2)<i) || (final{n+2}(to_check_a(c),b+3)<i) || (final{n+2}(to_check_a(c),b+4)<i) || (final{n+2}(to_check_a(c),b+5)<i) || (final{n+3}(to_check_a(c),b-5)<i) || (final{n+3}(to_check_a(c),b-4)<i) || (final{n+3}(to_check_a(c),b-3)<i) || (final{n+3}(to_check_a(c),b-2)<i) || (final{n+3}(to_check_a(c),b-1)<i) || (final{n+3}(to_check_a(c),b)<i) || (final{n+3}(to_check_a(c),b+1)<i) || (final{n+3}(to_check_a(c),b+2)<i) || (final{n+3}(to_check_a(c),b+3)<i) || (final{n+3}(to_check_a(c),b+4)<i) || (final{n+3}(to_check_a(c),b+5)<i) || (final{n+4}(to_check_a(c),b-5)<i) || (final{n+4}(to_check_a(c),b-4)<i) || (final{n+4}(to_check_a(c),b-3)<i) || (final{n+4}(to_check_a(c),b-2)<i) || (final{n+4}(to_check_a(c),b-1)<i) || (final{n+4}(to_check_a(c),b)<i) || (final{n+4}(to_check_a(c),b+1)<i) || (final{n+4}(to_check_a(c),b+2)<i) || (final{n+4}(to_check_a(c),b+3)<i) || (final{n+4}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i) || (final{n+5}(to_check_a(c),b-5)<i) || (final{n+5}(to_check_a(c),b-4)<i) || (final{n+5}(to_check_a(c),b-3)<i) || (final{n+5}(to_check_a(c),b-2)<i) || (final{n+5}(to_check_a(c),b-1)<i) || (final{n+5}(to_check_a(c),b)<i) || (final{n+5}(to_check_a(c),b+1)<i) || (final{n+5}(to_check_a(c),b+2)<i) || (final{n+5}(to_check_a(c),b+3)<i) || (final{n+5}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i)
            %                             detected="N";
            %                             break
            %                         end
            %                     end
            %                     if detected=="Y"
            %                         Y{m,i} = n;
            %                         Yp{m,i}=[a,b];
            %                         break
            %                     end
            %end

            %%for 10 x 10
            %              a=5;
            %              while a <= imgsize{m}(1)-5
            %                   to_check_a=[a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4, a+5];
            %                   for b = 5:imgsize{m}(2)-5
            %                       detected="Y";
            %                       for c = 1:10
            %                           if (final{n}(to_check_a(c),b-4)<i) || (final{n}(to_check_a(c),b-3)<i) || (final{n}(to_check_a(c),b-2)<i) || (final{n}(to_check_a(c),b-1)<i) || (final{n}(to_check_a(c),b)<i) || (final{n}(to_check_a(c),b+1)<i) || (final{n}(to_check_a(c),b+2)<i) || (final{n}(to_check_a(c),b+3)<i) || (final{n}(to_check_a(c),b+4)<i) || (final{n}(to_check_a(c),b+5)<i) || (final{n+1}(to_check_a(c),b-4)<i) || (final{n+1}(to_check_a(c),b-3)<i) || (final{n+1}(to_check_a(c),b-2)<i) || (final{n+1}(to_check_a(c),b-1)<i) || (final{n+1}(to_check_a(c),b)<i) || (final{n+1}(to_check_a(c),b+1)<i) || (final{n+1}(to_check_a(c),b+2)<i) || (final{n+1}(to_check_a(c),b+3)<i) || (final{n+1}(to_check_a(c),b+4)<i) || (final{n+1}(to_check_a(c),b+5)<i) || (final{n+2}(to_check_a(c),b-4)<i) || (final{n+2}(to_check_a(c),b-3)<i) || (final{n+2}(to_check_a(c),b-2)<i) || (final{n+2}(to_check_a(c),b-1)<i) || (final{n+2}(to_check_a(c),b)<i) || (final{n+2}(to_check_a(c),b+1)<i) || (final{n+2}(to_check_a(c),b+2)<i) || (final{n+2}(to_check_a(c),b+3)<i) || (final{n+2}(to_check_a(c),b+4)<i) || (final{n+2}(to_check_a(c),b+5)<i) || (final{n+3}(to_check_a(c),b-4)<i) || (final{n+3}(to_check_a(c),b-3)<i) || (final{n+3}(to_check_a(c),b-2)<i) || (final{n+3}(to_check_a(c),b-1)<i) || (final{n+3}(to_check_a(c),b)<i) || (final{n+3}(to_check_a(c),b+1)<i) || (final{n+3}(to_check_a(c),b+2)<i) || (final{n+3}(to_check_a(c),b+3)<i) || (final{n+3}(to_check_a(c),b+4)<i) || (final{n+3}(to_check_a(c),b+5)<i) || (final{n+4}(to_check_a(c),b-4)<i) || (final{n+4}(to_check_a(c),b-3)<i) || (final{n+4}(to_check_a(c),b-2)<i) || (final{n+4}(to_check_a(c),b-1)<i) || (final{n+4}(to_check_a(c),b)<i) || (final{n+4}(to_check_a(c),b+1)<i) || (final{n+4}(to_check_a(c),b+2)<i) || (final{n+4}(to_check_a(c),b+3)<i) || (final{n+4}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i) || (final{n+5}(to_check_a(c),b-4)<i) || (final{n+5}(to_check_a(c),b-3)<i) || (final{n+5}(to_check_a(c),b-2)<i) || (final{n+5}(to_check_a(c),b-1)<i) || (final{n+5}(to_check_a(c),b)<i) || (final{n+5}(to_check_a(c),b+1)<i) || (final{n+5}(to_check_a(c),b+2)<i) || (final{n+5}(to_check_a(c),b+3)<i) || (final{n+5}(to_check_a(c),b+4)<i) || (final{n+5}(to_check_a(c),b+5)<i)
            %                               detected="N";
            %                               break
            %                           end
            %                       end
            %                       if detected=="Y"
            %                           Y{m,i} = n;
            %                           Yp{m,i}=[a,b];
            %                           break
            %                       end
            %                   end

            %%for 9 x 9
            %             a=5;
            %             while a <= imgsize{m}(1)-4
            %                 for b = 5:imgsize{m}(2)-4
            %                         if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) && (final{n}(a-1,b-1)>=i) && (final{n}(a+2,b)>=i) && (final{n}(a-2,b)>=i) && (final{n}(a,b+2)>=i) && (final{n}(a,b-2)>=i) && (final{n}(a+2,b+2)>=i) && (final{n}(a-2,b+2)>=i) && (final{n}(a+2,b-2)>=i) && (final{n}(a-2,b-2)>=i) && (final{n}(a+1,b+2)>=i) && (final{n}(a-1,b+2)>=i) && (final{n}(a+1,b-2)>=i) && (final{n}(a-1,b-2)>=i) && (final{n}(a+2,b+1)>=i) && (final{n}(a+2,b-1)>=i) && (final{n}(a-2,b+1)>=i) && (final{n}(a-2,b-1)>=i) && (final{n}(a-3,b-3)>=i) && (final{n}(a-3,b-2)>=i) && (final{n}(a-3,b-1)>=i) && (final{n}(a-3,b)>=i) && (final{n}(a-3,b+1)>=i) && (final{n}(a-3,b+2)>=i) && (final{n}(a-3,b+3)>=i) && (final{n}(a-2,b+3)>=i) && (final{n}(a-1,b+3)>=i) && (final{n}(a,b+3)>=i) && (final{n}(a+1,b+3)>=i) && (final{n}(a+2,b+3)>=i) && (final{n}(a+3,b+3)>=i) && (final{n}(a+3,b+2)>=i) && (final{n}(a+3,b+1)>=i) && (final{n}(a+3,b)>=i) && (final{n}(a+3,b-1)>=i) && (final{n}(a+3,b-2)>=i) && (final{n}(a+3,b-3)>=i) && (final{n}(a+2,b-3)>=i) && (final{n}(a+1,b-3)>=i) && (final{n}(a,b-3)>=i) && (final{n}(a-1,b-3)>=i) && (final{n}(a-2,b-3)>=i) && (final{n}(a-4,b-4)>=i) && (final{n}(a-4,b-3)>=i) && (final{n}(a-4,b-2)>=i) && (final{n}(a-4,b-1)>=i) && (final{n}(a-4,b)>=i) && (final{n}(a-4,b+1)>=i) && (final{n}(a-4,b+2)>=i) && (final{n}(a-4,b+3)>=i) && (final{n}(a-4,b+4)>=i) && (final{n}(a-3,b+4)>=i) && (final{n}(a-2,b+4)>=i) && (final{n}(a-1,b+4)>=i) && (final{n}(a,b+4)>=i) && (final{n}(a+1,b+4)>=i) && (final{n}(a+2,b+4)>=i) && (final{n}(a+3,b+4)>=i) && (final{n}(a+4,b+4)>=i) && (final{n}(a+4,b+3)>=i) && (final{n}(a+4,b+2)>=i) && (final{n}(a+4,b+1)>=i) && (final{n}(a+4,b)>=i) && (final{n}(a+4,b-1)>=i) && (final{n}(a+4,b-2)>=i) && (final{n}(a+4,b-3)>=i) && (final{n}(a+4,b-4)>=i) && (final{n}(a+3,b-4)>=i) && (final{n}(a+2,b-4)>=i) && (final{n}(a+1,b-4)>=i) && (final{n}(a,b-4)>=i) && (final{n}(a-1,b-4)>=i) && (final{n}(a-2,b-4)>=i) && (final{n}(a-3,b-4)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) && (final{n+1}(a-1,b-1)>=i) && (final{n+1}(a+2,b)>=i) && (final{n+1}(a-2,b)>=i) && (final{n+1}(a,b+2)>=i) && (final{n+1}(a,b-2)>=i) && (final{n+1}(a+2,b+2)>=i) && (final{n+1}(a-2,b+2)>=i) && (final{n+1}(a+2,b-2)>=i) && (final{n+1}(a-2,b-2)>=i) && (final{n+1}(a+1,b+2)>=i) && (final{n+1}(a-1,b+2)>=i) && (final{n+1}(a+1,b-2)>=i) && (final{n+1}(a-1,b-2)>=i) && (final{n+1}(a+2,b+1)>=i) && (final{n+1}(a+2,b-1)>=i) && (final{n+1}(a-2,b+1)>=i) && (final{n+1}(a-2,b-1)>=i) && (final{n+1}(a-3,b-3)>=i) && (final{n+1}(a-3,b-2)>=i) && (final{n+1}(a-3,b-1)>=i) && (final{n+1}(a-3,b)>=i) && (final{n+1}(a-3,b+1)>=i) && (final{n+1}(a-3,b+2)>=i) && (final{n+1}(a-3,b+3)>=i) && (final{n+1}(a-2,b+3)>=i) && (final{n+1}(a-1,b+3)>=i) && (final{n+1}(a,b+3)>=i) && (final{n+1}(a+1,b+3)>=i) && (final{n+1}(a+2,b+3)>=i) && (final{n+1}(a+3,b+3)>=i) && (final{n+1}(a+3,b+2)>=i) && (final{n+1}(a+3,b+1)>=i) && (final{n+1}(a+3,b)>=i) && (final{n+1}(a+3,b-1)>=i) && (final{n+1}(a+3,b-2)>=i) && (final{n+1}(a+3,b-3)>=i) && (final{n+1}(a+2,b-3)>=i) && (final{n+1}(a+1,b-3)>=i) && (final{n+1}(a,b-3)>=i) && (final{n+1}(a-1,b-3)>=i) && (final{n+1}(a-2,b-3)>=i) && (final{n+1}(a-4,b-4)>=i) && (final{n+1}(a-4,b-3)>=i) && (final{n+1}(a-4,b-2)>=i) && (final{n+1}(a-4,b-1)>=i) && (final{n+1}(a-4,b)>=i) && (final{n+1}(a-4,b+1)>=i) && (final{n+1}(a-4,b+2)>=i) && (final{n+1}(a-4,b+3)>=i) && (final{n+1}(a-4,b+4)>=i) && (final{n+1}(a-3,b+4)>=i) && (final{n+1}(a-2,b+4)>=i) && (final{n+1}(a-1,b+4)>=i) && (final{n+1}(a,b+4)>=i) && (final{n+1}(a+1,b+4)>=i) && (final{n+1}(a+2,b+4)>=i) && (final{n+1}(a+3,b+4)>=i) && (final{n+1}(a+4,b+4)>=i) && (final{n+1}(a+4,b+3)>=i) && (final{n+1}(a+4,b+2)>=i) && (final{n+1}(a+4,b+1)>=i) && (final{n+1}(a+4,b)>=i) && (final{n+1}(a+4,b-1)>=i) && (final{n+1}(a+4,b-2)>=i) && (final{n+1}(a+4,b-3)>=i) && (final{n+1}(a+4,b-4)>=i) && (final{n+1}(a+3,b-4)>=i) && (final{n+1}(a+2,b-4)>=i) && (final{n+1}(a+1,b-4)>=i) && (final{n+1}(a,b-4)>=i) && (final{n+1}(a-1,b-4)>=i) && (final{n+1}(a-2,b-4)>=i) && (final{n+1}(a-3,b-4)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) && (final{n+2}(a-1,b-1)>=i) && (final{n+2}(a+2,b)>=i) && (final{n+2}(a-2,b)>=i) && (final{n+2}(a,b+2)>=i) && (final{n+2}(a,b-2)>=i) && (final{n+2}(a+2,b+2)>=i) && (final{n+2}(a-2,b+2)>=i) && (final{n+2}(a+2,b-2)>=i) && (final{n+2}(a-2,b-2)>=i) && (final{n+2}(a+1,b+2)>=i) && (final{n+2}(a-1,b+2)>=i) && (final{n+2}(a+1,b-2)>=i) && (final{n+2}(a-1,b-2)>=i) && (final{n+2}(a+2,b+1)>=i) && (final{n+2}(a+2,b-1)>=i) && (final{n+2}(a-2,b+1)>=i) && (final{n+2}(a-2,b-1)>=i) && (final{n+2}(a-3,b-3)>=i) && (final{n+2}(a-3,b-2)>=i) && (final{n+2}(a-3,b-1)>=i) && (final{n+2}(a-3,b)>=i) && (final{n+2}(a-3,b+1)>=i) && (final{n+2}(a-3,b+2)>=i) && (final{n+2}(a-3,b+3)>=i) && (final{n+2}(a-2,b+3)>=i) && (final{n+2}(a-1,b+3)>=i) && (final{n+2}(a,b+3)>=i) && (final{n+2}(a+1,b+3)>=i) && (final{n+2}(a+2,b+3)>=i) && (final{n+2}(a+3,b+3)>=i) && (final{n+2}(a+3,b+2)>=i) && (final{n+2}(a+3,b+1)>=i) && (final{n+2}(a+3,b)>=i) && (final{n+2}(a+3,b-1)>=i) && (final{n+2}(a+3,b-2)>=i) && (final{n+2}(a+3,b-3)>=i) && (final{n+2}(a+2,b-3)>=i) && (final{n+2}(a+1,b-3)>=i) && (final{n+2}(a,b-3)>=i) && (final{n+2}(a-1,b-3)>=i) && (final{n+2}(a-2,b-3)>=i) && (final{n+2}(a-4,b-4)>=i) && (final{n+2}(a-4,b-3)>=i) && (final{n+2}(a-4,b-2)>=i) && (final{n+2}(a-4,b-1)>=i) && (final{n+2}(a-4,b)>=i) && (final{n+2}(a-4,b+1)>=i) && (final{n+2}(a-4,b+2)>=i) && (final{n+2}(a-4,b+3)>=i) && (final{n+2}(a-4,b+4)>=i) && (final{n+2}(a-3,b+4)>=i) && (final{n+2}(a-2,b+4)>=i) && (final{n+2}(a-1,b+4)>=i) && (final{n+2}(a,b+4)>=i) && (final{n+2}(a+1,b+4)>=i) && (final{n+2}(a+2,b+4)>=i) && (final{n+2}(a+3,b+4)>=i) && (final{n+2}(a+4,b+4)>=i) && (final{n+2}(a+4,b+3)>=i) && (final{n+2}(a+4,b+2)>=i) && (final{n+2}(a+4,b+1)>=i) && (final{n+2}(a+4,b)>=i) && (final{n+2}(a+4,b-1)>=i) && (final{n+2}(a+4,b-2)>=i) && (final{n+2}(a+4,b-3)>=i) && (final{n+2}(a+4,b-4)>=i) && (final{n+2}(a+3,b-4)>=i) && (final{n+2}(a+2,b-4)>=i) && (final{n+2}(a+1,b-4)>=i) && (final{n+2}(a,b-4)>=i) && (final{n+2}(a-1,b-4)>=i) && (final{n+2}(a-2,b-4)>=i) && (final{n+2}(a-3,b-4)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) && (final{n+3}(a-1,b-1)>=i) && (final{n+3}(a+2,b)>=i) && (final{n+3}(a-2,b)>=i) && (final{n+3}(a,b+2)>=i) && (final{n+3}(a,b-2)>=i) && (final{n+3}(a+2,b+2)>=i) && (final{n+3}(a-2,b+2)>=i) && (final{n+3}(a+2,b-2)>=i) && (final{n+3}(a-2,b-2)>=i) && (final{n+3}(a+1,b+2)>=i) && (final{n+3}(a-1,b+2)>=i) && (final{n+3}(a+1,b-2)>=i) && (final{n+3}(a-1,b-2)>=i) && (final{n+3}(a+2,b+1)>=i) && (final{n+3}(a+2,b-1)>=i) && (final{n+3}(a-2,b+1)>=i) && (final{n+3}(a-2,b-1)>=i) && (final{n+3}(a-3,b-3)>=i) && (final{n+3}(a-3,b-2)>=i) && (final{n+3}(a-3,b-1)>=i) && (final{n+3}(a-3,b)>=i) && (final{n+3}(a-3,b+1)>=i) && (final{n+3}(a-3,b+2)>=i) && (final{n+3}(a-3,b+3)>=i) && (final{n+3}(a-2,b+3)>=i) && (final{n+3}(a-1,b+3)>=i) && (final{n+3}(a,b+3)>=i) && (final{n+3}(a+1,b+3)>=i) && (final{n+3}(a+2,b+3)>=i) && (final{n+3}(a+3,b+3)>=i) && (final{n+3}(a+3,b+2)>=i) && (final{n+3}(a+3,b+1)>=i) && (final{n+3}(a+3,b)>=i) && (final{n+3}(a+3,b-1)>=i) && (final{n+3}(a+3,b-2)>=i) && (final{n+3}(a+3,b-3)>=i) && (final{n+3}(a+2,b-3)>=i) && (final{n+3}(a+1,b-3)>=i) && (final{n+3}(a,b-3)>=i) && (final{n+3}(a-1,b-3)>=i) && (final{n+3}(a-2,b-3)>=i) && (final{n+3}(a-4,b-4)>=i) && (final{n+3}(a-4,b-3)>=i) && (final{n+3}(a-4,b-2)>=i) && (final{n+3}(a-4,b-1)>=i) && (final{n+3}(a-4,b)>=i) && (final{n+3}(a-4,b+1)>=i) && (final{n+3}(a-4,b+2)>=i) && (final{n+3}(a-4,b+3)>=i) && (final{n+3}(a-4,b+4)>=i) && (final{n+3}(a-3,b+4)>=i) && (final{n+3}(a-2,b+4)>=i) && (final{n+3}(a-1,b+4)>=i) && (final{n+3}(a,b+4)>=i) && (final{n+3}(a+1,b+4)>=i) && (final{n+3}(a+2,b+4)>=i) && (final{n+3}(a+3,b+4)>=i) && (final{n+3}(a+4,b+4)>=i) && (final{n+3}(a+4,b+3)>=i) && (final{n+3}(a+4,b+2)>=i) && (final{n+3}(a+4,b+1)>=i) && (final{n+3}(a+4,b)>=i) && (final{n+3}(a+4,b-1)>=i) && (final{n+3}(a+4,b-2)>=i) && (final{n+3}(a+4,b-3)>=i) && (final{n+3}(a+4,b-4)>=i) && (final{n+3}(a+3,b-4)>=i) && (final{n+3}(a+2,b-4)>=i) && (final{n+3}(a+1,b-4)>=i) && (final{n+3}(a,b-4)>=i) && (final{n+3}(a-1,b-4)>=i) && (final{n+3}(a-2,b-4)>=i) && (final{n+3}(a-3,b-4)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) && (final{n+4}(a-1,b-1)>=i) && (final{n+4}(a+2,b)>=i) && (final{n+4}(a-2,b)>=i) && (final{n+4}(a,b+2)>=i) && (final{n+4}(a,b-2)>=i) && (final{n+4}(a+2,b+2)>=i) && (final{n+4}(a-2,b+2)>=i) && (final{n+4}(a+2,b-2)>=i) && (final{n+4}(a-2,b-2)>=i) && (final{n+4}(a+1,b+2)>=i) && (final{n+4}(a-1,b+2)>=i) && (final{n+4}(a+1,b-2)>=i) && (final{n+4}(a-1,b-2)>=i) && (final{n+4}(a+2,b+1)>=i) && (final{n+4}(a+2,b-1)>=i) && (final{n+4}(a-2,b+1)>=i) && (final{n+4}(a-2,b-1)>=i) && (final{n+4}(a-3,b-3)>=i) && (final{n+4}(a-3,b-2)>=i) && (final{n+4}(a-3,b-1)>=i) && (final{n+4}(a-3,b)>=i) && (final{n+4}(a-3,b+1)>=i) && (final{n+4}(a-3,b+2)>=i) && (final{n+4}(a-3,b+3)>=i) && (final{n+4}(a-2,b+3)>=i) && (final{n+4}(a-1,b+3)>=i) && (final{n+4}(a,b+3)>=i) && (final{n+4}(a+1,b+3)>=i) && (final{n+4}(a+2,b+3)>=i) && (final{n+4}(a+3,b+3)>=i) && (final{n+4}(a+3,b+2)>=i) && (final{n+4}(a+3,b+1)>=i) && (final{n+4}(a+3,b)>=i) && (final{n+4}(a+3,b-1)>=i) && (final{n+4}(a+3,b-2)>=i) && (final{n+4}(a+3,b-3)>=i) && (final{n+4}(a+2,b-3)>=i) && (final{n+4}(a+1,b-3)>=i) && (final{n+4}(a,b-3)>=i) && (final{n+4}(a-1,b-3)>=i) && (final{n+4}(a-2,b-3)>=i) && (final{n+4}(a-4,b-4)>=i) && (final{n+4}(a-4,b-3)>=i) && (final{n+4}(a-4,b-2)>=i) && (final{n+4}(a-4,b-1)>=i) && (final{n+4}(a-4,b)>=i) && (final{n+4}(a-4,b+1)>=i) && (final{n+4}(a-4,b+2)>=i) && (final{n+4}(a-4,b+3)>=i) && (final{n+4}(a-4,b+4)>=i) && (final{n+4}(a-3,b+4)>=i) && (final{n+4}(a-2,b+4)>=i) && (final{n+4}(a-1,b+4)>=i) && (final{n+4}(a,b+4)>=i) && (final{n+4}(a+1,b+4)>=i) && (final{n+4}(a+2,b+4)>=i) && (final{n+4}(a+3,b+4)>=i) && (final{n+4}(a+4,b+4)>=i) && (final{n+4}(a+4,b+3)>=i) && (final{n+4}(a+4,b+2)>=i) && (final{n+4}(a+4,b+1)>=i) && (final{n+4}(a+4,b)>=i) && (final{n+4}(a+4,b-1)>=i) && (final{n+4}(a+4,b-2)>=i) && (final{n+4}(a+4,b-3)>=i) && (final{n+4}(a+4,b-4)>=i) && (final{n+4}(a+3,b-4)>=i) && (final{n+4}(a+2,b-4)>=i) && (final{n+4}(a+1,b-4)>=i) && (final{n+4}(a,b-4)>=i) && (final{n+4}(a-1,b-4)>=i) && (final{n+4}(a-2,b-4)>=i) && (final{n+4}(a-3,b-4)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) && (final{n+5}(a-1,b-1)>=i) && (final{n+5}(a+2,b)>=i) && (final{n+5}(a-2,b)>=i) && (final{n+5}(a,b+2)>=i) && (final{n+5}(a,b-2)>=i) && (final{n+5}(a+2,b+2)>=i) && (final{n+5}(a-2,b+2)>=i) && (final{n+5}(a+2,b-2)>=i) && (final{n+5}(a-2,b-2)>=i) && (final{n+5}(a+1,b+2)>=i) && (final{n+5}(a-1,b+2)>=i) && (final{n+5}(a+1,b-2)>=i) && (final{n+5}(a-1,b-2)>=i) && (final{n+5}(a+2,b+1)>=i) && (final{n+5}(a+2,b-1)>=i) && (final{n+5}(a-2,b+1)>=i) && (final{n+5}(a-2,b-1)>=i) && (final{n+5}(a-3,b-3)>=i) && (final{n+5}(a-3,b-2)>=i) && (final{n+5}(a-3,b-1)>=i) && (final{n+5}(a-3,b)>=i) && (final{n+5}(a-3,b+1)>=i) && (final{n+5}(a-3,b+2)>=i) && (final{n+5}(a-3,b+3)>=i) && (final{n+5}(a-2,b+3)>=i) && (final{n+5}(a-1,b+3)>=i) && (final{n+5}(a,b+3)>=i) && (final{n+5}(a+1,b+3)>=i) && (final{n+5}(a+2,b+3)>=i) && (final{n+5}(a+3,b+3)>=i) && (final{n+5}(a+3,b+2)>=i) && (final{n+5}(a+3,b+1)>=i) && (final{n+5}(a+3,b)>=i) && (final{n+5}(a+3,b-1)>=i) && (final{n+5}(a+3,b-2)>=i) && (final{n+5}(a+3,b-3)>=i) && (final{n+5}(a+2,b-3)>=i) && (final{n+5}(a+1,b-3)>=i) && (final{n+5}(a,b-3)>=i) && (final{n+5}(a-1,b-3)>=i) && (final{n+5}(a-2,b-3)>=i) && (final{n+5}(a-4,b-4)>=i) && (final{n+5}(a-4,b-3)>=i) && (final{n+5}(a-4,b-2)>=i) && (final{n+5}(a-4,b-1)>=i) && (final{n+5}(a-4,b)>=i) && (final{n+5}(a-4,b+1)>=i) && (final{n+5}(a-4,b+2)>=i) && (final{n+5}(a-4,b+3)>=i) && (final{n+5}(a-4,b+4)>=i) && (final{n+5}(a-3,b+4)>=i) && (final{n+5}(a-2,b+4)>=i) && (final{n+5}(a-1,b+4)>=i) && (final{n+5}(a,b+4)>=i) && (final{n+5}(a+1,b+4)>=i) && (final{n+5}(a+2,b+4)>=i) && (final{n+5}(a+3,b+4)>=i) && (final{n+5}(a+4,b+4)>=i) && (final{n+5}(a+4,b+3)>=i) && (final{n+5}(a+4,b+2)>=i) && (final{n+5}(a+4,b+1)>=i) && (final{n+5}(a+4,b)>=i) && (final{n+5}(a+4,b-1)>=i) && (final{n+5}(a+4,b-2)>=i) && (final{n+5}(a+4,b-3)>=i) && (final{n+5}(a+4,b-4)>=i) && (final{n+5}(a+3,b-4)>=i) && (final{n+5}(a+2,b-4)>=i) && (final{n+5}(a+1,b-4)>=i) && (final{n+5}(a,b-4)>=i) && (final{n+5}(a-1,b-4)>=i) && (final{n+5}(a-2,b-4)>=i) && (final{n+5}(a-3,b-4)>=i)
            %                          Y{m,i} = n;
            %                          Yp{m,i}=[a,b];
            %                          break
            %                         end
            %                 end

            %%for 8 x 8
            %               a=4;
            %               while a <= imgsize{m}(1)-4
            %                     for b = 4:imgsize{m}(2)-4
            %                             if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) && (final{n}(a-1,b-1)>=i) && (final{n}(a+2,b)>=i) && (final{n}(a-2,b)>=i) && (final{n}(a,b+2)>=i) && (final{n}(a,b-2)>=i) && (final{n}(a+2,b+2)>=i) && (final{n}(a-2,b+2)>=i) && (final{n}(a+2,b-2)>=i) && (final{n}(a-2,b-2)>=i) && (final{n}(a+1,b+2)>=i) && (final{n}(a-1,b+2)>=i) && (final{n}(a+1,b-2)>=i) && (final{n}(a-1,b-2)>=i) && (final{n}(a+2,b+1)>=i) && (final{n}(a+2,b-1)>=i) && (final{n}(a-2,b+1)>=i) && (final{n}(a-2,b-1)>=i) && (final{n}(a-3,b-3)>=i) && (final{n}(a-3,b-2)>=i) && (final{n}(a-3,b-1)>=i) && (final{n}(a-3,b)>=i) && (final{n}(a-3,b+1)>=i) && (final{n}(a-3,b+2)>=i) && (final{n}(a-3,b+3)>=i) && (final{n}(a-2,b+3)>=i) && (final{n}(a-1,b+3)>=i) && (final{n}(a,b+3)>=i) && (final{n}(a+1,b+3)>=i) && (final{n}(a+2,b+3)>=i) && (final{n}(a+3,b+3)>=i) && (final{n}(a+3,b+2)>=i) && (final{n}(a+3,b+1)>=i) && (final{n}(a+3,b)>=i) && (final{n}(a+3,b-1)>=i) && (final{n}(a+3,b-2)>=i) && (final{n}(a+3,b-3)>=i) && (final{n}(a+2,b-3)>=i) && (final{n}(a+1,b-3)>=i) && (final{n}(a,b-3)>=i) && (final{n}(a-1,b-3)>=i) && (final{n}(a-2,b-3)>=i) && (final{n}(a-3,b+4)>=i) && (final{n}(a-2,b+4)>=i) && (final{n}(a-1,b+4)>=i) && (final{n}(a,b+4)>=i) && (final{n}(a+1,b+4)>=i) && (final{n}(a+2,b+4)>=i) && (final{n}(a+3,b+4)>=i) && (final{n}(a+4,b+4)>=i) && (final{n}(a+4,b+3)>=i) && (final{n}(a+4,b+2)>=i) && (final{n}(a+4,b+1)>=i) && (final{n}(a+4,b)>=i) && (final{n}(a+4,b-1)>=i) && (final{n}(a+4,b-2)>=i) && (final{n}(a+4,b-3)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) && (final{n+1}(a-1,b-1)>=i) && (final{n+1}(a+2,b)>=i) && (final{n+1}(a-2,b)>=i) && (final{n+1}(a,b+2)>=i) && (final{n+1}(a,b-2)>=i) && (final{n+1}(a+2,b+2)>=i) && (final{n+1}(a-2,b+2)>=i) && (final{n+1}(a+2,b-2)>=i) && (final{n+1}(a-2,b-2)>=i) && (final{n+1}(a+1,b+2)>=i) && (final{n+1}(a-1,b+2)>=i) && (final{n+1}(a+1,b-2)>=i) && (final{n+1}(a-1,b-2)>=i) && (final{n+1}(a+2,b+1)>=i) && (final{n+1}(a+2,b-1)>=i) && (final{n+1}(a-2,b+1)>=i) && (final{n+1}(a-2,b-1)>=i) && (final{n+1}(a-3,b-3)>=i) && (final{n+1}(a-3,b-2)>=i) && (final{n+1}(a-3,b-1)>=i) && (final{n+1}(a-3,b)>=i) && (final{n+1}(a-3,b+1)>=i) && (final{n+1}(a-3,b+2)>=i) && (final{n+1}(a-3,b+3)>=i) && (final{n+1}(a-2,b+3)>=i) && (final{n+1}(a-1,b+3)>=i) && (final{n+1}(a,b+3)>=i) && (final{n+1}(a+1,b+3)>=i) && (final{n+1}(a+2,b+3)>=i) && (final{n+1}(a+3,b+3)>=i) && (final{n+1}(a+3,b+2)>=i) && (final{n+1}(a+3,b+1)>=i) && (final{n+1}(a+3,b)>=i) && (final{n+1}(a+3,b-1)>=i) && (final{n+1}(a+3,b-2)>=i) && (final{n+1}(a+3,b-3)>=i) && (final{n+1}(a+2,b-3)>=i) && (final{n+1}(a+1,b-3)>=i) && (final{n+1}(a,b-3)>=i) && (final{n+1}(a-1,b-3)>=i) && (final{n+1}(a-2,b-3)>=i) && (final{n+1}(a-3,b+4)>=i) && (final{n+1}(a-2,b+4)>=i) && (final{n+1}(a-1,b+4)>=i) && (final{n+1}(a,b+4)>=i) && (final{n+1}(a+1,b+4)>=i) && (final{n+1}(a+2,b+4)>=i) && (final{n+1}(a+3,b+4)>=i) && (final{n+1}(a+4,b+4)>=i) && (final{n+1}(a+4,b+3)>=i) && (final{n+1}(a+4,b+2)>=i) && (final{n+1}(a+4,b+1)>=i) && (final{n+1}(a+4,b)>=i) && (final{n+1}(a+4,b-1)>=i) && (final{n+1}(a+4,b-2)>=i) && (final{n+1}(a+4,b-3)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) && (final{n+2}(a-1,b-1)>=i) && (final{n+2}(a+2,b)>=i) && (final{n+2}(a-2,b)>=i) && (final{n+2}(a,b+2)>=i) && (final{n+2}(a,b-2)>=i) && (final{n+2}(a+2,b+2)>=i) && (final{n+2}(a-2,b+2)>=i) && (final{n+2}(a+2,b-2)>=i) && (final{n+2}(a-2,b-2)>=i) && (final{n+2}(a+1,b+2)>=i) && (final{n+2}(a-1,b+2)>=i) && (final{n+2}(a+1,b-2)>=i) && (final{n+2}(a-1,b-2)>=i) && (final{n+2}(a+2,b+1)>=i) && (final{n+2}(a+2,b-1)>=i) && (final{n+2}(a-2,b+1)>=i) && (final{n+2}(a-2,b-1)>=i) && (final{n+2}(a-3,b-3)>=i) && (final{n+2}(a-3,b-2)>=i) && (final{n+2}(a-3,b-1)>=i) && (final{n+2}(a-3,b)>=i) && (final{n+2}(a-3,b+1)>=i) && (final{n+2}(a-3,b+2)>=i) && (final{n+2}(a-3,b+3)>=i) && (final{n+2}(a-2,b+3)>=i) && (final{n+2}(a-1,b+3)>=i) && (final{n+2}(a,b+3)>=i) && (final{n+2}(a+1,b+3)>=i) && (final{n+2}(a+2,b+3)>=i) && (final{n+2}(a+3,b+3)>=i) && (final{n+2}(a+3,b+2)>=i) && (final{n+2}(a+3,b+1)>=i) && (final{n+2}(a+3,b)>=i) && (final{n+2}(a+3,b-1)>=i) && (final{n+2}(a+3,b-2)>=i) && (final{n+2}(a+3,b-3)>=i) && (final{n+2}(a+2,b-3)>=i) && (final{n+2}(a+1,b-3)>=i) && (final{n+2}(a,b-3)>=i) && (final{n+2}(a-1,b-3)>=i) && (final{n+2}(a-2,b-3)>=i) && (final{n+2}(a-3,b+4)>=i) && (final{n+2}(a-2,b+4)>=i) && (final{n+2}(a-1,b+4)>=i) && (final{n+2}(a,b+4)>=i) && (final{n+2}(a+1,b+4)>=i) && (final{n+2}(a+2,b+4)>=i) && (final{n+2}(a+3,b+4)>=i) && (final{n+2}(a+4,b+4)>=i) && (final{n+2}(a+4,b+3)>=i) && (final{n+2}(a+4,b+2)>=i) && (final{n+2}(a+4,b+1)>=i) && (final{n+2}(a+4,b)>=i) && (final{n+2}(a+4,b-1)>=i) && (final{n+2}(a+4,b-2)>=i) && (final{n+2}(a+4,b-3)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) && (final{n+3}(a-1,b-1)>=i) && (final{n+3}(a+2,b)>=i) && (final{n+3}(a-2,b)>=i) && (final{n+3}(a,b+2)>=i) && (final{n+3}(a,b-2)>=i) && (final{n+3}(a+2,b+2)>=i) && (final{n+3}(a-2,b+2)>=i) && (final{n+3}(a+2,b-2)>=i) && (final{n+3}(a-2,b-2)>=i) && (final{n+3}(a+1,b+2)>=i) && (final{n+3}(a-1,b+2)>=i) && (final{n+3}(a+1,b-2)>=i) && (final{n+3}(a-1,b-2)>=i) && (final{n+3}(a+2,b+1)>=i) && (final{n+3}(a+2,b-1)>=i) && (final{n+3}(a-2,b+1)>=i) && (final{n+3}(a-2,b-1)>=i) && (final{n+3}(a-3,b-3)>=i) && (final{n+3}(a-3,b-2)>=i) && (final{n+3}(a-3,b-1)>=i) && (final{n+3}(a-3,b)>=i) && (final{n+3}(a-3,b+1)>=i) && (final{n+3}(a-3,b+2)>=i) && (final{n+3}(a-3,b+3)>=i) && (final{n+3}(a-2,b+3)>=i) && (final{n+3}(a-1,b+3)>=i) && (final{n+3}(a,b+3)>=i) && (final{n+3}(a+1,b+3)>=i) && (final{n+3}(a+2,b+3)>=i) && (final{n+3}(a+3,b+3)>=i) && (final{n+3}(a+3,b+2)>=i) && (final{n+3}(a+3,b+1)>=i) && (final{n+3}(a+3,b)>=i) && (final{n+3}(a+3,b-1)>=i) && (final{n+3}(a+3,b-2)>=i) && (final{n+3}(a+3,b-3)>=i) && (final{n+3}(a+2,b-3)>=i) && (final{n+3}(a+1,b-3)>=i) && (final{n+3}(a,b-3)>=i) && (final{n+3}(a-1,b-3)>=i) && (final{n+3}(a-2,b-3)>=i) && (final{n+3}(a-3,b+4)>=i) && (final{n+3}(a-2,b+4)>=i) && (final{n+3}(a-1,b+4)>=i) && (final{n+3}(a,b+4)>=i) && (final{n+3}(a+1,b+4)>=i) && (final{n+3}(a+2,b+4)>=i) && (final{n+3}(a+3,b+4)>=i) && (final{n+3}(a+4,b+4)>=i) && (final{n+3}(a+4,b+3)>=i) && (final{n+3}(a+4,b+2)>=i) && (final{n+3}(a+4,b+1)>=i) && (final{n+3}(a+4,b)>=i) && (final{n+3}(a+4,b-1)>=i) && (final{n+3}(a+4,b-2)>=i) && (final{n+3}(a+4,b-3)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) && (final{n+4}(a-1,b-1)>=i) && (final{n+4}(a+2,b)>=i) && (final{n+4}(a-2,b)>=i) && (final{n+4}(a,b+2)>=i) && (final{n+4}(a,b-2)>=i) && (final{n+4}(a+2,b+2)>=i) && (final{n+4}(a-2,b+2)>=i) && (final{n+4}(a+2,b-2)>=i) && (final{n+4}(a-2,b-2)>=i) && (final{n+4}(a+1,b+2)>=i) && (final{n+4}(a-1,b+2)>=i) && (final{n+4}(a+1,b-2)>=i) && (final{n+4}(a-1,b-2)>=i) && (final{n+4}(a+2,b+1)>=i) && (final{n+4}(a+2,b-1)>=i) && (final{n+4}(a-2,b+1)>=i) && (final{n+4}(a-2,b-1)>=i) && (final{n+4}(a-3,b-3)>=i) && (final{n+4}(a-3,b-2)>=i) && (final{n+4}(a-3,b-1)>=i) && (final{n+4}(a-3,b)>=i) && (final{n+4}(a-3,b+1)>=i) && (final{n+4}(a-3,b+2)>=i) && (final{n+4}(a-3,b+3)>=i) && (final{n+4}(a-2,b+3)>=i) && (final{n+4}(a-1,b+3)>=i) && (final{n+4}(a,b+3)>=i) && (final{n+4}(a+1,b+3)>=i) && (final{n+4}(a+2,b+3)>=i) && (final{n+4}(a+3,b+3)>=i) && (final{n+4}(a+3,b+2)>=i) && (final{n+4}(a+3,b+1)>=i) && (final{n+4}(a+3,b)>=i) && (final{n+4}(a+3,b-1)>=i) && (final{n+4}(a+3,b-2)>=i) && (final{n+4}(a+3,b-3)>=i) && (final{n+4}(a+2,b-3)>=i) && (final{n+4}(a+1,b-3)>=i) && (final{n+4}(a,b-3)>=i) && (final{n+4}(a-1,b-3)>=i) && (final{n+4}(a-2,b-3)>=i) && (final{n+4}(a-3,b+4)>=i) && (final{n+4}(a-2,b+4)>=i) && (final{n+4}(a-1,b+4)>=i) && (final{n+4}(a,b+4)>=i) && (final{n+4}(a+1,b+4)>=i) && (final{n+4}(a+2,b+4)>=i) && (final{n+4}(a+3,b+4)>=i) && (final{n+4}(a+4,b+4)>=i) && (final{n+4}(a+4,b+3)>=i) && (final{n+4}(a+4,b+2)>=i) && (final{n+4}(a+4,b+1)>=i) && (final{n+4}(a+4,b)>=i) && (final{n+4}(a+4,b-1)>=i) && (final{n+4}(a+4,b-2)>=i) && (final{n+4}(a+4,b-3)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) && (final{n+5}(a-1,b-1)>=i) && (final{n+5}(a+2,b)>=i) && (final{n+5}(a-2,b)>=i) && (final{n+5}(a,b+2)>=i) && (final{n+5}(a,b-2)>=i) && (final{n+5}(a+2,b+2)>=i) && (final{n+5}(a-2,b+2)>=i) && (final{n+5}(a+2,b-2)>=i) && (final{n+5}(a-2,b-2)>=i) && (final{n+5}(a+1,b+2)>=i) && (final{n+5}(a-1,b+2)>=i) && (final{n+5}(a+1,b-2)>=i) && (final{n+5}(a-1,b-2)>=i) && (final{n+5}(a+2,b+1)>=i) && (final{n+5}(a+2,b-1)>=i) && (final{n+5}(a-2,b+1)>=i) && (final{n+5}(a-2,b-1)>=i) && (final{n+5}(a-3,b-3)>=i) && (final{n+5}(a-3,b-2)>=i) && (final{n+5}(a-3,b-1)>=i) && (final{n+5}(a-3,b)>=i) && (final{n+5}(a-3,b+1)>=i) && (final{n+5}(a-3,b+2)>=i) && (final{n+5}(a-3,b+3)>=i) && (final{n+5}(a-2,b+3)>=i) && (final{n+5}(a-1,b+3)>=i) && (final{n+5}(a,b+3)>=i) && (final{n+5}(a+1,b+3)>=i) && (final{n+5}(a+2,b+3)>=i) && (final{n+5}(a+3,b+3)>=i) && (final{n+5}(a+3,b+2)>=i) && (final{n+5}(a+3,b+1)>=i) && (final{n+5}(a+3,b)>=i) && (final{n+5}(a+3,b-1)>=i) && (final{n+5}(a+3,b-2)>=i) && (final{n+5}(a+3,b-3)>=i) && (final{n+5}(a+2,b-3)>=i) && (final{n+5}(a+1,b-3)>=i) && (final{n+5}(a,b-3)>=i) && (final{n+5}(a-1,b-3)>=i) && (final{n+5}(a-2,b-3)>=i) && (final{n+5}(a-3,b+4)>=i) && (final{n+5}(a-2,b+4)>=i) && (final{n+5}(a-1,b+4)>=i) && (final{n+5}(a,b+4)>=i) && (final{n+5}(a+1,b+4)>=i) && (final{n+5}(a+2,b+4)>=i) && (final{n+5}(a+3,b+4)>=i) && (final{n+5}(a+4,b+4)>=i) && (final{n+5}(a+4,b+3)>=i) && (final{n+5}(a+4,b+2)>=i) && (final{n+5}(a+4,b+1)>=i) && (final{n+5}(a+4,b)>=i) && (final{n+5}(a+4,b-1)>=i) && (final{n+5}(a+4,b-2)>=i) && (final{n+5}(a+4,b-3)>=i)
            %                              Y{m,i} = n;
            %                              Yp{m,i}=[a,b];
            %                              break
            %                             end
            %                     end

            %%for 7 x 7
            %             a=4;
            %             while a <= imgsize{m}(1)-3
            %                 for b = 4:imgsize{m}(2)-3
            %                         if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) &&(final{n}(a-1,b-1)>=i) && (final{n}(a+2,b)>=i) && (final{n}(a-2,b)>=i) && (final{n}(a,b+2)>=i) && (final{n}(a,b-2)>=i) && (final{n}(a+2,b+2)>=i) && (final{n}(a-2,b+2)>=i) && (final{n}(a+2,b-2)>=i) && (final{n}(a-2,b-2)>=i) && (final{n}(a+1,b+2)>=i) && (final{n}(a-1,b+2)>=i) && (final{n}(a+1,b-2)>=i) && (final{n}(a-1,b-2)>=i) && (final{n}(a+2,b+1)>=i) && (final{n}(a+2,b-1)>=i) && (final{n}(a-2,b+1)>=i) && (final{n}(a-2,b-1)>=i) && (final{n}(a-3,b-3)>=i) && (final{n}(a-3,b-2)>=i) && (final{n}(a-3,b-1)>=i) && (final{n}(a-3,b)>=i) && (final{n}(a-3,b+1)>=i) && (final{n}(a-3,b+2)>=i) && (final{n}(a-3,b+3)>=i) && (final{n}(a-2,b+3)>=i) && (final{n}(a-1,b+3)>=i) && (final{n}(a,b+3)>=i) && (final{n}(a+1,b+3)>=i) && (final{n}(a+2,b+3)>=i) && (final{n}(a+3,b+3)>=i) && (final{n}(a+3,b+2)>=i) && (final{n}(a+3,b+1)>=i) && (final{n}(a+3,b)>=i) && (final{n}(a+3,b-1)>=i) && (final{n}(a+3,b-2)>=i) && (final{n}(a+3,b-3)>=i) && (final{n}(a+2,b-3)>=i) && (final{n}(a+1,b-3)>=i) && (final{n}(a,b-3)>=i) && (final{n}(a-1,b-3)>=i) && (final{n}(a-2,b-3)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) &&(final{n+1}(a-1,b-1)>=i) && (final{n+1}(a+2,b)>=i) && (final{n+1}(a-2,b)>=i) && (final{n+1}(a,b+2)>=i) && (final{n+1}(a,b-2)>=i) && (final{n+1}(a+2,b+2)>=i) && (final{n+1}(a-2,b+2)>=i) && (final{n+1}(a+2,b-2)>=i) && (final{n+1}(a-2,b-2)>=i) && (final{n+1}(a+1,b+2)>=i) && (final{n+1}(a-1,b+2)>=i) && (final{n+1}(a+1,b-2)>=i) && (final{n+1}(a-1,b-2)>=i) && (final{n+1}(a+2,b+1)>=i) && (final{n+1}(a+2,b-1)>=i) && (final{n+1}(a-2,b+1)>=i) && (final{n+1}(a-2,b-1)>=i) && (final{n+1}(a-3,b-3)>=i) && (final{n+1}(a-3,b-2)>=i) && (final{n+1}(a-3,b-1)>=i) && (final{n+1}(a-3,b)>=i) && (final{n+1}(a-3,b+1)>=i) && (final{n+1}(a-3,b+2)>=i) && (final{n+1}(a-3,b+3)>=i) && (final{n+1}(a-2,b+3)>=i) && (final{n+1}(a-1,b+3)>=i) && (final{n+1}(a,b+3)>=i) && (final{n+1}(a+1,b+3)>=i) && (final{n+1}(a+2,b+3)>=i) && (final{n+1}(a+3,b+3)>=i) && (final{n+1}(a+3,b+2)>=i) && (final{n+1}(a+3,b+1)>=i) && (final{n+1}(a+3,b)>=i) && (final{n+1}(a+3,b-1)>=i) && (final{n+1}(a+3,b-2)>=i) && (final{n+1}(a+3,b-3)>=i) && (final{n+1}(a+2,b-3)>=i) && (final{n+1}(a+1,b-3)>=i) && (final{n+1}(a,b-3)>=i) && (final{n+1}(a-1,b-3)>=i) && (final{n+1}(a-2,b-3)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) &&(final{n+2}(a-1,b-1)>=i) && (final{n+2}(a+2,b)>=i) && (final{n+2}(a-2,b)>=i) && (final{n+2}(a,b+2)>=i) && (final{n+2}(a,b-2)>=i) && (final{n+2}(a+2,b+2)>=i) && (final{n+2}(a-2,b+2)>=i) && (final{n+2}(a+2,b-2)>=i) && (final{n+2}(a-2,b-2)>=i) && (final{n+2}(a+1,b+2)>=i) && (final{n+2}(a-1,b+2)>=i) && (final{n+2}(a+1,b-2)>=i) && (final{n+2}(a-1,b-2)>=i) && (final{n+2}(a+2,b+1)>=i) && (final{n+2}(a+2,b-1)>=i) && (final{n+2}(a-2,b+1)>=i) && (final{n+2}(a-2,b-1)>=i) && (final{n+2}(a-3,b-3)>=i) && (final{n+2}(a-3,b-2)>=i) && (final{n+2}(a-3,b-1)>=i) && (final{n+2}(a-3,b)>=i) && (final{n+2}(a-3,b+1)>=i) && (final{n+2}(a-3,b+2)>=i) && (final{n+2}(a-3,b+3)>=i) && (final{n+2}(a-2,b+3)>=i) && (final{n+2}(a-1,b+3)>=i) && (final{n+2}(a,b+3)>=i) && (final{n+2}(a+1,b+3)>=i) && (final{n+2}(a+2,b+3)>=i) && (final{n+2}(a+3,b+3)>=i) && (final{n+2}(a+3,b+2)>=i) && (final{n+2}(a+3,b+1)>=i) && (final{n+2}(a+3,b)>=i) && (final{n+2}(a+3,b-1)>=i) && (final{n+2}(a+3,b-2)>=i) && (final{n+2}(a+3,b-3)>=i) && (final{n+2}(a+2,b-3)>=i) && (final{n+2}(a+1,b-3)>=i) && (final{n+2}(a,b-3)>=i) && (final{n+2}(a-1,b-3)>=i) && (final{n+2}(a-2,b-3)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) &&(final{n+3}(a-1,b-1)>=i) && (final{n+3}(a+2,b)>=i) && (final{n+3}(a-2,b)>=i) && (final{n+3}(a,b+2)>=i) && (final{n+3}(a,b-2)>=i) && (final{n+3}(a+2,b+2)>=i) && (final{n+3}(a-2,b+2)>=i) && (final{n+3}(a+2,b-2)>=i) && (final{n+3}(a-2,b-2)>=i) && (final{n+3}(a+1,b+2)>=i) && (final{n+3}(a-1,b+2)>=i) && (final{n+3}(a+1,b-2)>=i) && (final{n+3}(a-1,b-2)>=i) && (final{n+3}(a+2,b+1)>=i) && (final{n+3}(a+2,b-1)>=i) && (final{n+3}(a-2,b+1)>=i) && (final{n+3}(a-2,b-1)>=i) && (final{n+3}(a-3,b-3)>=i) && (final{n+3}(a-3,b-2)>=i) && (final{n+3}(a-3,b-1)>=i) && (final{n+3}(a-3,b)>=i) && (final{n+3}(a-3,b+1)>=i) && (final{n+3}(a-3,b+2)>=i) && (final{n+3}(a-3,b+3)>=i) && (final{n+3}(a-2,b+3)>=i) && (final{n+3}(a-1,b+3)>=i) && (final{n+3}(a,b+3)>=i) && (final{n+3}(a+1,b+3)>=i) && (final{n+3}(a+2,b+3)>=i) && (final{n+3}(a+3,b+3)>=i) && (final{n+3}(a+3,b+2)>=i) && (final{n+3}(a+3,b+1)>=i) && (final{n+3}(a+3,b)>=i) && (final{n+3}(a+3,b-1)>=i) && (final{n+3}(a+3,b-2)>=i) && (final{n+3}(a+3,b-3)>=i) && (final{n+3}(a+2,b-3)>=i) && (final{n+3}(a+1,b-3)>=i) && (final{n+3}(a,b-3)>=i) && (final{n+3}(a-1,b-3)>=i) && (final{n+3}(a-2,b-3)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) &&(final{n+4}(a-1,b-1)>=i) && (final{n+4}(a+2,b)>=i) && (final{n+4}(a-2,b)>=i) && (final{n+4}(a,b+2)>=i) && (final{n+4}(a,b-2)>=i) && (final{n+4}(a+2,b+2)>=i) && (final{n+4}(a-2,b+2)>=i) && (final{n+4}(a+2,b-2)>=i) && (final{n+4}(a-2,b-2)>=i) && (final{n+4}(a+1,b+2)>=i) && (final{n+4}(a-1,b+2)>=i) && (final{n+4}(a+1,b-2)>=i) && (final{n+4}(a-1,b-2)>=i) && (final{n+4}(a+2,b+1)>=i) && (final{n+4}(a+2,b-1)>=i) && (final{n+4}(a-2,b+1)>=i) && (final{n+4}(a-2,b-1)>=i) && (final{n+4}(a-3,b-3)>=i) && (final{n+4}(a-3,b-2)>=i) && (final{n+4}(a-3,b-1)>=i) && (final{n+4}(a-3,b)>=i) && (final{n+4}(a-3,b+1)>=i) && (final{n+4}(a-3,b+2)>=i) && (final{n+4}(a-3,b+3)>=i) && (final{n+4}(a-2,b+3)>=i) && (final{n+4}(a-1,b+3)>=i) && (final{n+4}(a,b+3)>=i) && (final{n+4}(a+1,b+3)>=i) && (final{n+4}(a+2,b+3)>=i) && (final{n+4}(a+3,b+3)>=i) && (final{n+4}(a+3,b+2)>=i) && (final{n+4}(a+3,b+1)>=i) && (final{n+4}(a+3,b)>=i) && (final{n+4}(a+3,b-1)>=i) && (final{n+4}(a+3,b-2)>=i) && (final{n+4}(a+3,b-3)>=i) && (final{n+4}(a+2,b-3)>=i) && (final{n+4}(a+1,b-3)>=i) && (final{n+4}(a,b-3)>=i) && (final{n+4}(a-1,b-3)>=i) && (final{n+4}(a-2,b-3)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) &&(final{n+5}(a-1,b-1)>=i) && (final{n+5}(a+2,b)>=i) && (final{n+5}(a-2,b)>=i) && (final{n+5}(a,b+2)>=i) && (final{n+5}(a,b-2)>=i) && (final{n+5}(a+2,b+2)>=i) && (final{n+5}(a-2,b+2)>=i) && (final{n+5}(a+2,b-2)>=i) && (final{n+5}(a-2,b-2)>=i) && (final{n+5}(a+1,b+2)>=i) && (final{n+5}(a-1,b+2)>=i) && (final{n+5}(a+1,b-2)>=i) && (final{n+5}(a-1,b-2)>=i) && (final{n+5}(a+2,b+1)>=i) && (final{n+5}(a+2,b-1)>=i) && (final{n+5}(a-2,b+1)>=i) && (final{n+5}(a-2,b-1)>=i) && (final{n+5}(a-3,b-3)>=i) && (final{n+5}(a-3,b-2)>=i) && (final{n+5}(a-3,b-1)>=i) && (final{n+5}(a-3,b)>=i) && (final{n+5}(a-3,b+1)>=i) && (final{n+5}(a-3,b+2)>=i) && (final{n+5}(a-3,b+3)>=i) && (final{n+5}(a-2,b+3)>=i) && (final{n+5}(a-1,b+3)>=i) && (final{n+5}(a,b+3)>=i) && (final{n+5}(a+1,b+3)>=i) && (final{n+5}(a+2,b+3)>=i) && (final{n+5}(a+3,b+3)>=i) && (final{n+5}(a+3,b+2)>=i) && (final{n+5}(a+3,b+1)>=i) && (final{n+5}(a+3,b)>=i) && (final{n+5}(a+3,b-1)>=i) && (final{n+5}(a+3,b-2)>=i) && (final{n+5}(a+3,b-3)>=i) && (final{n+5}(a+2,b-3)>=i) && (final{n+5}(a+1,b-3)>=i) && (final{n+5}(a,b-3)>=i) && (final{n+5}(a-1,b-3) >=i) && (final{n+5}(a-2,b-3)>=i)
            %                          Y{m,i} = n;
            %                          Yp{m,i}=[a,b];
            %                          break
            %                         end
            %                 end

            %%for 6 x 6
            %               a=3;
            %               while a <= imgsize{m}(1)-3
            %                      for b = 3:imgsize{m}(2)-3
            %                              if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) &&(final{n}(a-1,b-1)>=i) && (final{n}(a+2,b)>=i) && (final{n}(a-2,b)>=i) && (final{n}(a,b+2)>=i) && (final{n}(a,b-2)>=i) && (final{n}(a+2,b+2)>=i) && (final{n}(a-2,b+2)>=i) && (final{n}(a+2,b-2)>=i) && (final{n}(a-2,b-2)>=i) && (final{n}(a+1,b+2)>=i) && (final{n}(a-1,b+2)>=i) && (final{n}(a+1,b-2)>=i) && (final{n}(a-1,b-2)>=i) && (final{n}(a+2,b+1)>=i) && (final{n}(a+2,b-1)>=i) && (final{n}(a-2,b+1)>=i) && (final{n}(a-2,b-1)>=i) && (final{n}(a-2,b+3)>=i) && (final{n}(a-1,b+3)>=i) && (final{n}(a,b+3)>=i) && (final{n}(a+1,b+3)>=i) && (final{n}(a+2,b+3)>=i) && (final{n}(a+3,b+3)>=i) && (final{n}(a+3,b+2)>=i) && (final{n}(a+3,b+1)>=i) && (final{n}(a+3,b)>=i) && (final{n}(a+3,b-1)>=i) && (final{n}(a+3,b-2)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) &&(final{n+1}(a-1,b-1)>=i) && (final{n+1}(a+2,b)>=i) && (final{n+1}(a-2,b)>=i) && (final{n+1}(a,b+2)>=i) && (final{n+1}(a,b-2)>=i) && (final{n+1}(a+2,b+2)>=i) && (final{n+1}(a-2,b+2)>=i) && (final{n+1}(a+2,b-2)>=i) && (final{n+1}(a-2,b-2)>=i) && (final{n+1}(a+1,b+2)>=i) && (final{n+1}(a-1,b+2)>=i) && (final{n+1}(a+1,b-2)>=i) && (final{n+1}(a-1,b-2)>=i) && (final{n+1}(a+2,b+1)>=i) && (final{n+1}(a+2,b-1)>=i) && (final{n+1}(a-2,b+1)>=i) && (final{n+1}(a-2,b-1)>=i) && (final{n+1}(a-2,b+3)>=i) && (final{n+1}(a-1,b+3)>=i) && (final{n+1}(a,b+3)>=i) && (final{n+1}(a+1,b+3)>=i) && (final{n+1}(a+2,b+3)>=i) && (final{n+1}(a+3,b+3)>=i) && (final{n+1}(a+3,b+2)>=i) && (final{n+1}(a+3,b+1)>=i) && (final{n+1}(a+3,b)>=i) && (final{n+1}(a+3,b-1)>=i) && (final{n+1}(a+3,b-2)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) &&(final{n+2}(a-1,b-1)>=i) && (final{n+2}(a+2,b)>=i) && (final{n+2}(a-2,b)>=i) && (final{n+2}(a,b+2)>=i) && (final{n+2}(a,b-2)>=i) && (final{n+2}(a+2,b+2)>=i) && (final{n+2}(a-2,b+2)>=i) && (final{n+2}(a+2,b-2)>=i) && (final{n+2}(a-2,b-2)>=i) && (final{n+2}(a+1,b+2)>=i) && (final{n+2}(a-1,b+2)>=i) && (final{n+2}(a+1,b-2)>=i) && (final{n+2}(a-1,b-2)>=i) && (final{n+2}(a+2,b+1)>=i) && (final{n+2}(a+2,b-1)>=i) && (final{n+2}(a-2,b+1)>=i) && (final{n+2}(a-2,b-1)>=i) && (final{n+2}(a-2,b+3)>=i) && (final{n+2}(a-1,b+3)>=i) && (final{n+2}(a,b+3)>=i) && (final{n+2}(a+1,b+3)>=i) && (final{n+2}(a+2,b+3)>=i) && (final{n+2}(a+3,b+3)>=i) && (final{n+2}(a+3,b+2)>=i) && (final{n+2}(a+3,b+1)>=i) && (final{n+2}(a+3,b)>=i) && (final{n+2}(a+3,b-1)>=i) && (final{n+2}(a+3,b-2)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) &&(final{n+3}(a-1,b-1)>=i) && (final{n+3}(a+2,b)>=i) && (final{n+3}(a-2,b)>=i) && (final{n+3}(a,b+2)>=i) && (final{n+3}(a,b-2)>=i) && (final{n+3}(a+2,b+2)>=i) && (final{n+3}(a-2,b+2)>=i) && (final{n+3}(a+2,b-2)>=i) && (final{n+3}(a-2,b-2)>=i) && (final{n+3}(a+1,b+2)>=i) && (final{n+3}(a-1,b+2)>=i) && (final{n+3}(a+1,b-2)>=i) && (final{n+3}(a-1,b-2)>=i) && (final{n+3}(a+2,b+1)>=i) && (final{n+3}(a+2,b-1)>=i) && (final{n+3}(a-2,b+1)>=i) && (final{n+3}(a-2,b-1)>=i) && (final{n+3}(a-2,b+3)>=i) && (final{n+3}(a-1,b+3)>=i) && (final{n+3}(a,b+3)>=i) && (final{n+3}(a+1,b+3)>=i) && (final{n+3}(a+2,b+3)>=i) && (final{n+3}(a+3,b+3)>=i) && (final{n+3}(a+3,b+2)>=i) && (final{n+3}(a+3,b+1)>=i) && (final{n+3}(a+3,b)>=i) && (final{n+3}(a+3,b-1)>=i) && (final{n+3}(a+3,b-2)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) &&(final{n+4}(a-1,b-1)>=i) && (final{n+4}(a+2,b)>=i) && (final{n+4}(a-2,b)>=i) && (final{n+4}(a,b+2)>=i) && (final{n+4}(a,b-2)>=i) && (final{n+4}(a+2,b+2)>=i) && (final{n+4}(a-2,b+2)>=i) && (final{n+4}(a+2,b-2)>=i) && (final{n+4}(a-2,b-2)>=i) && (final{n+4}(a+1,b+2)>=i) && (final{n+4}(a-1,b+2)>=i) && (final{n+4}(a+1,b-2)>=i) && (final{n+4}(a-1,b-2)>=i) && (final{n+4}(a+2,b+1)>=i) && (final{n+4}(a+2,b-1)>=i) && (final{n+4}(a-2,b+1)>=i) && (final{n+4}(a-2,b-1)>=i) && (final{n+4}(a-2,b+3)>=i) && (final{n+4}(a-1,b+3)>=i) && (final{n+4}(a,b+3)>=i) && (final{n+4}(a+1,b+3)>=i) && (final{n+4}(a+2,b+3)>=i) && (final{n+4}(a+3,b+3)>=i) && (final{n+4}(a+3,b+2)>=i) && (final{n+4}(a+3,b+1)>=i) && (final{n+4}(a+3,b)>=i) && (final{n+4}(a+3,b-1)>=i) && (final{n+4}(a+3,b-2)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) &&(final{n+5}(a-1,b-1)>=i) && (final{n+5}(a+2,b)>=i) && (final{n+5}(a-2,b)>=i) && (final{n+5}(a,b+2)>=i) && (final{n+5}(a,b-2)>=i) && (final{n+5}(a+2,b+2)>=i) && (final{n+5}(a-2,b+2)>=i) && (final{n+5}(a+2,b-2)>=i) && (final{n+5}(a-2,b-2)>=i) && (final{n+5}(a+1,b+2)>=i) && (final{n+5}(a-1,b+2)>=i) && (final{n+5}(a+1,b-2)>=i) && (final{n+5}(a-1,b-2)>=i) && (final{n+5}(a+2,b+1)>=i) && (final{n+5}(a+2,b-1)>=i) && (final{n+5}(a-2,b+1)>=i) && (final{n+5}(a-2,b-1)>=i) && (final{n+5}(a-2,b+3)>=i) && (final{n+5}(a-1,b+3)>=i) && (final{n+5}(a,b+3)>=i) && (final{n+5}(a+1,b+3)>=i) && (final{n+5}(a+2,b+3)>=i) && (final{n+5}(a+3,b+3)>=i) && (final{n+5}(a+3,b+2)>=i) && (final{n+5}(a+3,b+1)>=i) && (final{n+5}(a+3,b)>=i) && (final{n+5}(a+3,b-1)>=i) && (final{n+5}(a+3,b-2)>=i)
            %                               Y{m,i} = n;
            %                               Yp{m,i}=[a,b];
            %                               break
            %                              end
            %                      end

            %%for 5 x 5
            a = 3;

            while a <= imgsize{m}(1) - 2

                for b = 3:imgsize{m}(2) - 2

                    if (final{n}(a, b) >= i) && (final{n}(a + 1, b) >= i) && (final{n}(a - 1, b) >= i) && (final{n}(a, b + 1) >= i) && (final{n}(a, b - 1) >= i) && (final{n}(a + 1, b + 1) >= i) && (final{n}(a + 1, b - 1) >= i) && (final{n}(a - 1, b + 1) >= i) && (final{n}(a - 1, b - 1) >= i) && (final{n}(a + 2, b) >= i) && (final{n}(a - 2, b) >= i) && (final{n}(a, b + 2) >= i) && (final{n}(a, b - 2) >= i) && (final{n}(a + 2, b + 2) >= i) && (final{n}(a - 2, b + 2) >= i) && (final{n}(a + 2, b - 2) >= i) && (final{n}(a - 2, b - 2) >= i) && (final{n}(a + 1, b + 2) >= i) && (final{n}(a - 1, b + 2) >= i) && (final{n}(a + 1, b - 2) >= i) && (final{n}(a - 1, b - 2) >= i) && (final{n}(a + 2, b + 1) >= i) && (final{n}(a + 2, b - 1) >= i) && (final{n}(a - 2, b + 1) >= i) && (final{n}(a - 2, b - 1) >= i) && (final{n + 1}(a, b) >= i) && (final{n + 1}(a + 1, b) >= i) && (final{n + 1}(a - 1, b) >= i) && (final{n + 1}(a, b + 1) >= i) && (final{n + 1}(a, b - 1) >= i) && (final{n + 1}(a + 1, b + 1) >= i) && (final{n + 1}(a + 1, b - 1) >= i) && (final{n + 1}(a - 1, b + 1) >= i) && (final{n + 1}(a - 1, b - 1) >= i) && (final{n + 1}(a + 2, b) >= i) && (final{n + 1}(a - 2, b) >= i) && (final{n + 1}(a, b + 2) >= i) && (final{n + 1}(a, b - 2) >= i) && (final{n + 1}(a + 2, b + 2) >= i) && (final{n + 1}(a - 2, b + 2) >= i) && (final{n + 1}(a + 2, b - 2) >= i) && (final{n + 1}(a - 2, b - 2) >= i) && (final{n + 1}(a + 1, b + 2) >= i) && (final{n + 1}(a - 1, b + 2) >= i) && (final{n + 1}(a + 1, b - 2) >= i) && (final{n + 1}(a - 1, b - 2) >= i) && (final{n + 1}(a + 2, b + 1) >= i) && (final{n + 1}(a + 2, b - 1) >= i) && (final{n + 1}(a - 2, b + 1) >= i) && (final{n + 1}(a - 2, b - 1) >= i) && (final{n + 2}(a, b) >= i) && (final{n + 2}(a + 1, b) >= i) && (final{n + 2}(a - 1, b) >= i) && (final{n + 2}(a, b + 1) >= i) && (final{n + 2}(a, b - 1) >= i) && (final{n + 2}(a + 1, b + 1) >= i) && (final{n + 2}(a + 1, b - 1) >= i) && (final{n + 2}(a - 1, b + 1) >= i) && (final{n + 2}(a - 1, b - 1) >= i) && (final{n + 2}(a + 2, b) >= i) && (final{n + 2}(a - 2, b) >= i) && (final{n + 2}(a, b + 2) >= i) && (final{n + 2}(a, b - 2) >= i) && (final{n + 2}(a + 2, b + 2) >= i) && (final{n + 2}(a - 2, b + 2) >= i) && (final{n + 2}(a + 2, b - 2) >= i) && (final{n + 2}(a - 2, b - 2) >= i) && (final{n + 2}(a + 1, b + 2) >= i) && (final{n + 2}(a - 1, b + 2) >= i) && (final{n + 2}(a + 1, b - 2) >= i) && (final{n + 2}(a - 1, b - 2) >= i) && (final{n + 2}(a + 2, b + 1) >= i) && (final{n + 2}(a + 2, b - 1) >= i) && (final{n + 2}(a - 2, b + 1) >= i) && (final{n + 2}(a - 2, b - 1) >= i) && (final{n + 3}(a, b) >= i) && (final{n + 3}(a + 1, b) >= i) && (final{n + 3}(a - 1, b) >= i) && (final{n + 3}(a, b + 1) >= i) && (final{n + 3}(a, b - 1) >= i) && (final{n + 3}(a + 1, b + 1) >= i) && (final{n + 3}(a + 1, b - 1) >= i) && (final{n + 3}(a - 1, b + 1) >= i) && (final{n + 3}(a - 1, b - 1) >= i) && (final{n + 3}(a + 2, b) >= i) && (final{n + 3}(a - 2, b) >= i) && (final{n + 3}(a, b + 2) >= i) && (final{n + 3}(a, b - 2) >= i) && (final{n + 3}(a + 2, b + 2) >= i) && (final{n + 3}(a - 2, b + 2) >= i) && (final{n + 3}(a + 2, b - 2) >= i) && (final{n + 3}(a - 2, b - 2) >= i) && (final{n + 3}(a + 1, b + 2) >= i) && (final{n + 3}(a - 1, b + 2) >= i) && (final{n + 3}(a + 1, b - 2) >= i) && (final{n + 3}(a - 1, b - 2) >= i) && (final{n + 3}(a + 2, b + 1) >= i) && (final{n + 3}(a + 2, b - 1) >= i) && (final{n + 3}(a - 2, b + 1) >= i) && (final{n + 3}(a - 2, b - 1) >= i) && (final{n + 4}(a, b) >= i) && (final{n + 4}(a + 1, b) >= i) && (final{n + 4}(a - 1, b) >= i) && (final{n + 4}(a, b + 1) >= i) && (final{n + 4}(a, b - 1) >= i) && (final{n + 4}(a + 1, b + 1) >= i) && (final{n + 4}(a + 1, b - 1) >= i) && (final{n + 4}(a - 1, b + 1) >= i) && (final{n + 4}(a - 1, b - 1) >= i) && (final{n + 4}(a + 2, b) >= i) && (final{n + 4}(a - 2, b) >= i) && (final{n + 4}(a, b + 2) >= i) && (final{n + 4}(a, b - 2) >= i) && (final{n + 4}(a + 2, b + 2) >= i) && (final{n + 4}(a - 2, b + 2) >= i) && (final{n + 4}(a + 2, b - 2) >= i) && (final{n + 4}(a - 2, b - 2) >= i) && (final{n + 4}(a + 1, b + 2) >= i) && (final{n + 4}(a - 1, b + 2) >= i) && (final{n + 4}(a + 1, b - 2) >= i) && (final{n + 4}(a - 1, b - 2) >= i) && (final{n + 4}(a + 2, b + 1) >= i) && (final{n + 4}(a + 2, b - 1) >= i) && (final{n + 4}(a - 2, b + 1) >= i) && (final{n + 4}(a - 2, b - 1) >= i) && (final{n + 5}(a, b) >= i) && (final{n + 5}(a + 1, b) >= i) && (final{n + 5}(a - 1, b) >= i) && (final{n + 5}(a, b + 1) >= i) && (final{n + 5}(a, b - 1) >= i) && (final{n + 5}(a + 1, b + 1) >= i) && (final{n + 5}(a + 1, b - 1) >= i) && (final{n + 5}(a - 1, b + 1) >= i) && (final{n + 5}(a - 1, b - 1) >= i) && (final{n + 5}(a + 2, b) >= i) && (final{n + 5}(a - 2, b) >= i) && (final{n + 5}(a, b + 2) >= i) && (final{n + 5}(a, b - 2) >= i) && (final{n + 5}(a + 2, b + 2) >= i) && (final{n + 5}(a - 2, b + 2) >= i) && (final{n + 5}(a + 2, b - 2) >= i) && (final{n + 5}(a - 2, b - 2) >= i) && (final{n + 5}(a + 1, b + 2) >= i) && (final{n + 5}(a - 1, b + 2) >= i) && (final{n + 5}(a + 1, b - 2) >= i) && (final{n + 5}(a - 1, b - 2) >= i) && (final{n + 5}(a + 2, b + 1) >= i) && (final{n + 5}(a + 2, b - 1) >= i) && (final{n + 5}(a - 2, b + 1) >= i) && (final{n + 5}(a - 2, b - 1) >= i)
                        Y{m, i} = n;
                        Yp{m, i} = [a, b];
                        break
                    end

                end

                %%for 4 x 4
                %               a=2;
                %               while a <= imgsize{m}(1)-2
                %                     for b = 2:imgsize{m}(2)-2
                %                            if (final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) && (final{n}(a-1,b-1)>=i) && (final{n}(a+2,b)>=i) && (final{n}(a,b+2)>=i) && (final{n}(a+2,b+2)>=i) && (final{n}(a+1,b+2)>=i) && (final{n}(a-1,b+2)>=i) && (final{n}(a+2,b+1)>=i) && (final{n}(a+2,b-1)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) && (final{n+1}(a-1,b-1)>=i) && (final{n+1}(a+2,b)>=i) && (final{n+1}(a,b+2)>=i) && (final{n+1}(a+2,b+2)>=i) && (final{n+1}(a+1,b+2)>=i) && (final{n+1}(a-1,b+2)>=i) && (final{n+1}(a+2,b+1)>=i) && (final{n+1}(a+2,b-1)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) && (final{n+2}(a-1,b-1)>=i) && (final{n+2}(a+2,b)>=i) && (final{n+2}(a,b+2)>=i) && (final{n+2}(a+2,b+2)>=i) && (final{n+2}(a+1,b+2)>=i) && (final{n+2}(a-1,b+2)>=i) && (final{n+2}(a+2,b+1)>=i) && (final{n+2}(a+2,b-1)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) && (final{n+3}(a-1,b-1)>=i) && (final{n+3}(a+2,b)>=i) && (final{n+3}(a,b+2)>=i) && (final{n+3}(a+2,b+2)>=i) && (final{n+3}(a+1,b+2)>=i) && (final{n+3}(a-1,b+2)>=i) && (final{n+3}(a+2,b+1)>=i) && (final{n+3}(a+2,b-1)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) && (final{n+4}(a-1,b-1)>=i) && (final{n+4}(a+2,b)>=i) && (final{n+4}(a,b+2)>=i) && (final{n+4}(a+2,b+2)>=i) && (final{n+4}(a+1,b+2)>=i) && (final{n+4}(a-1,b+2)>=i) && (final{n+4}(a+2,b+1)>=i) && (final{n+4}(a+2,b-1)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) && (final{n+5}(a-1,b-1)>=i) && (final{n+5}(a+2,b)>=i) && (final{n+5}(a,b+2)>=i) && (final{n+5}(a+2,b+2)>=i) && (final{n+5}(a+1,b+2)>=i) && (final{n+5}(a-1,b+2)>=i) && (final{n+5}(a+2,b+1)>=i) && (final{n+5}(a+2,b-1)>=i)
                %                             Y{m,i} = n;
                %                             Yp{m,i}=[a,b];
                %                             break
                %                            end
                %                     end

                %%for 3 x 3
                %             a=2;
                %             while a <= imgsize{m}(1)-1
                %                 for b = 2:imgsize{m}(2)-1
                %                        if(final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a-1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a,b-1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n}(a+1,b-1)>=i) && (final{n}(a-1,b+1)>=i) && (final{n}(a-1,b-1)>=i) &&(final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a-1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a,b-1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+1}(a+1,b-1)>=i) && (final{n+1}(a-1,b+1)>=i) && (final{n+1}(a-1,b-1)>=i) &&(final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a-1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a,b-1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+2}(a+1,b-1)>=i) && (final{n+2}(a-1,b+1)>=i) && (final{n+2}(a-1,b-1)>=i) &&(final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a-1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a,b-1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+3}(a+1,b-1)>=i) && (final{n+3}(a-1,b+1)>=i) && (final{n+3}(a-1,b-1)>=i) &&(final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a-1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a,b-1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+4}(a+1,b-1)>=i) && (final{n+4}(a-1,b+1)>=i) && (final{n+4}(a-1,b-1)>=i) &&(final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a-1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a,b-1)>=i) && (final{n+5}(a+1,b+1)>=i) && (final{n+5}(a+1,b-1)>=i) && (final{n+5}(a-1,b+1)>=i) && (final{n+5}(a-1,b-1)>=i)
                %                         Y{m,i} = n;
                %                         Yp{m,i}=[a,b];
                %                         break
                %                        end
                %                 end

                %%for 2 x 2
                %             a=1;
                %             while a <= imgsize{m}(1)-1
                %                 for b = 1:imgsize{m}(2)-1
                %                        if(final{n}(a,b)>=i) && (final{n}(a+1,b)>=i) && (final{n}(a,b+1)>=i) && (final{n}(a+1,b+1)>=i) && (final{n+1}(a,b)>=i) && (final{n+1}(a+1,b)>=i) && (final{n+1}(a,b+1)>=i) && (final{n+1}(a+1,b+1)>=i) && (final{n+2}(a,b)>=i) && (final{n+2}(a+1,b)>=i) && (final{n+2}(a,b+1)>=i) && (final{n+2}(a+1,b+1)>=i) && (final{n+3}(a,b)>=i) && (final{n+3}(a+1,b)>=i) && (final{n+3}(a,b+1)>=i) && (final{n+3}(a+1,b+1)>=i) && (final{n+4}(a,b)>=i) && (final{n+4}(a+1,b)>=i) && (final{n+4}(a,b+1)>=i) && (final{n+4}(a+1,b+1)>=i) && (final{n+5}(a,b)>=i) && (final{n+5}(a+1,b)>=i) && (final{n+5}(a,b+1)>=i) && (final{n+5}(a+1,b+1)>=i)
                %                         Y{m,i} = n;
                %                         Yp{m,i}=[a,b];
                %                         break
                %                        end
                %                 end

                if Y{m, i} == 0
                    a = a + 1;
                else
                    break
                end

            end

            % if no snowball found in data set, assigns default values of
            % largest image # in data set for Y position, [0, 0] for Yp
            % position
            if (Y{m, i} == 0) && (n == list2{m}(listsize{m} - 5))
                Y{m, i} = B(m);
                Yp{m, i} = [0, 0];
                break
            elseif Y{m, i} ~= 0
                break
            end

        end

    end

end

%%
% reads answer key in, subtracts each of its entries from those in the same
% row as it, puts the results in Z, and takes the average and uncertainty
% for each column
fileID = fopen('/Users/harrij17/Dropbox/SULI/Snowball/SNOWBALL CROPPED IMAGES/fiesta front w Be 10 - 8 bit/knownfiestafrontwBe10.txt', 'r');
C = fscanf(fileID, '%f');
fclose(fileID);
Z = zeros(eventsize(2), 255);

for m = 1:eventsize(2)

    for i = 1:255
        Z(m, i) = Y{m, i} - C(m);
    end

end

average = mean(Z);
uncertainty = std(Z) / sqrt(eventsize(2));

% calculates the amount of checks saved with lines 95-98 and the
% percentage of them (measures effect of lines 95-98)
eventsaved = 0;
eventtotal = 0;

for m = 1:eventsize(2)

    for i = 1:255
        eventtotal = eventtotal + 1;

        if i ~= 1 && Y{m, i} == B(m) && Y{m, i - 1} == B(m)
            eventsaved = eventsaved + 1;
        end

    end

end

percenteventsaved = eventsaved / eventtotal * 100;

% writes cell for the result file
for i = 1:255
    result{i, :, :, :} = [i, average(i), 0, uncertainty(i)];
end

%%
% saves main result file, snowball locations, raw snowball frames, and
% subtracted snowball frames
writecell(result, resultfile, 'Delimiter', 'tab')
writecell(Yp, comblocation)
writecell(Y, combframes, 'Delimiter', 'tab')
writematrix(Z, combframesZ, 'Delimiter', 'tab')
save(names{sizenames})
