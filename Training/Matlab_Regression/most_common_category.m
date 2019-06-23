TrainingY = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/TrainingDataY.csv');
TestingY = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/TestingDataY.csv');

DAS28 = TrainingY;
%%
remission = 0;
low = 0;
moderate = 0;
high = 0;

for i = 1:length(DAS28)
    if (DAS28(i) < 2.6)
        remission = remission + 1;
    elseif (DAS28(i) < 3.2)
        low = low + 1;
    elseif (DAS28(i) < 5.1)
        moderate = moderate + 1;
    else
        high = high + 1;
    end    
end
correct = 0;
for i = 1:length(TestingY)
    if TestingY(i) < 5.1 && TestingY(i) >= 3.2
        correct = correct + 1;
    end
end
correct / length(TestingY)