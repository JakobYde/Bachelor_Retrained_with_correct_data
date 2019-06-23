%% Starting data
DAS28_train = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/TrainingDataY.csv');
DAS28_test = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/TestingDataY.csv');

DAS28_mean = mean(DAS28_train)

Error_random_MAE = mean(abs(DAS28_mean - DAS28_test))

Error_random_MSE = mean((DAS28_mean - DAS28_test).^2)

%% Additional data

DAS28_train = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/TrainingDataY.csv');
additional_data = csvread('D:/WindowsFolders/Code/Data/BachelorFixedData/csv_files/NewDataRegression.csv');

DAS28_test = additional_data((1:2:length(additional_data(:,3))),3);

DAS28_mean = mean(DAS28_train)

Error_random_MAE = mean(abs(DAS28_mean - DAS28_test))

Error_random_MSE = mean((DAS28_mean - DAS28_test).^2)