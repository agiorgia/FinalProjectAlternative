%% Generate Dataset and Visualize
clear
rng(1); 
n_samples_train = 500;
n_samples_test = 5000;
r0 = 2;
r1 = 4;

x0_train = generate_class_samples(n_samples_train, r0);
x1_train = generate_class_samples(n_samples_train, r1);

x0_test = generate_class_samples(n_samples_test, r0);
x1_test = generate_class_samples(n_samples_test, r1);

% Oragnize training data and test data
X_train = zeros(length(x0_train)*2,2);
X_train(:,1) = [x0_train(:,1);x1_train(:,1)];
X_train(:,2) = [x0_train(:,2);x1_train(:,2)];
Y_train = [-ones(length(x0_train), 1); ones(length(x1_train), 1)]; % Labels

X_test = zeros(length(x0_test)*2,2);
X_test(:,1) = [x0_test(:,1);x1_test(:,1)];
X_test(:,2) = [x0_test(:,2);x1_test(:,2)];
Y_test = [-ones(length(x0_test), 1); ones(length(x1_test), 1)]; % Labels 

% Plot data
figure
subplot(1,2,1)
scatter(x0_train(:,1),x0_train(:,2));
hold on 
scatter(x1_train(:,1),x1_train(:,2));
hold off
title('Training Data Scatter Plot')
legend('Class -1','Class 1')

subplot(1,2,2)
scatter(x0_test(:,1),x0_test(:,2));
hold on 
scatter(x1_test(:,1),x1_test(:,2));
hold off
title('Test Data Scatter Plot')
legend('Class -1','Class 1')

% Plot data
figure
subplot(1,2,1)
scatter(x0_test(:,1),x0_test(:,2));
hold on 
viscircles([0 0],2)
hold off
title('Class -1 Test Data')
xlim([-8 8])
ylim([-8 8])

subplot(1,2,2)
scatter(10,10)
hold on 
scatter(x1_test(:,1),x1_test(:,2));
viscircles([0 0],4)
hold off
title('Class 1 Test Data')
xlim([-8 8])
ylim([-8 8])

%% SVM
rng(1)
% 1. Hyperparameter Tuning for SVM
%Use 10-fold cross-validation to find optimal box constraint and kernel width
cv = cvpartition(Y_train, 'KFold', 10);

% Define hyperparameter ranges
% boxConstraints = logspace(-3, 3, 5);
% kernelWidths = logspace(-1, 1, 5);

boxConstraints = 27:2:35;
kernelWidths = 1:2:9;

bestErrorSVM = Inf;
bestBox = 0;
bestKernelWidth = 0;

meanErrorArray = [];
bind = 0;
for box = boxConstraints
    bind = bind +1;
    kind = 0;
    for kernelWidth = kernelWidths
        kind = kind +1;
        errors = zeros(cv.NumTestSets, 1);
        for i = 1:cv.NumTestSets
            trainIdx = training(cv, i);
            testIdx = test(cv, i);
            SVMModel = fitcsvm(X_train(trainIdx, :), Y_train(trainIdx), ...
                'KernelFunction', 'gaussian', 'KernelScale', kernelWidth, ...
                'BoxConstraint', box, 'Standardize', true);
            predictions = predict(SVMModel, X_train(testIdx, :));
            errors(i) = mean(predictions ~= Y_train(testIdx));
        end
        meanError = mean(errors);
        meanErrorArray(kind,bind) = meanError;
        if meanError < bestErrorSVM
            bestErrorSVM = meanError;
            bestBox = box;
            bestKernelWidth = kernelWidth;
        end
    end
end


% Train the final SVM model with the optimal hyperparameters
finalSVMModel = fitcsvm(X_train, Y_train, ...
    'KernelFunction', 'gaussian', 'KernelScale', bestKernelWidth, ...
    'BoxConstraint', bestBox, 'Standardize', true);

% Test the final SVM model
SVM_predictions = predict(finalSVMModel, X_test);
SVM_error = mean(SVM_predictions ~= Y_test);

% Visualize SVM decision boundary
figure;
scatter(x0_test(:,1),x0_test(:,2));
hold on 
scatter(x1_test(:,1),x1_test(:,2));
sv = finalSVMModel.SupportVectors;
plot(sv(:, 1), sv(:, 2), 'k*');
title('SVM Decision Boundary and Test Data');
hold off;
legend('Class -1','Class 1','Boundary')

% Display Results 
disp('SVM Results:');
disp(['Optimal Box Constraint: ', num2str(bestBox)]);
disp(['Optimal Kernel Width: ', num2str(bestKernelWidth)]);
disp(['SVM Test Error: ', num2str(SVM_error)]);

%% MLP
rng(1)
% 2. Hyperparameter Tuning for MLP
% Define hyperparameter range for the number of perceptrons
%numPerceptronsRange = 1:10:41;
numPerceptronsRange = 19:23;
bestErrorMLP = Inf;
bestNumPerceptrons = 0;
arrayind = 0;

for numPerceptrons = numPerceptronsRange
    arrayind = arrayind+1;
    errors = zeros(cv.NumTestSets, 1);
    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        net = fitnet(numPerceptrons);
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.3;
        net.divideParam.testRatio = 0.0;
        net = train(net, X_train(trainIdx, :)', Y_train(trainIdx)');
        predictions = net(X_train(testIdx, :)');
        predictions = sign(predictions);
        errors(i) = mean(predictions ~= Y_train(testIdx)');
    end
    meanError = mean(errors);
    MLP_meanErrors(arrayind) = meanError;
    if meanError < bestErrorMLP
        bestErrorMLP = meanError;
        bestNumPerceptrons = numPerceptrons;
    end
end

% Train the final MLP model with the optimal number of perceptrons
finalNet = fitnet(bestNumPerceptrons);
finalNet = train(finalNet, X_train', Y_train');
MLP_predictions = finalNet(X_test');
MLP_predictions = sign(MLP_predictions);
MLP_error = mean(MLP_predictions ~= Y_test');

% Display Results
disp('MLP Results:');
disp(['Optimal Number of Perceptrons: ', num2str(bestNumPerceptrons)]);
disp(['MLP Test Error: ', num2str(MLP_error)]);

% Generate a grid of points for decision boundary
x1Range = linspace(min(X_test(:, 1)) - 1, max(X_test(:, 1)) + 1, 200);
x2Range = linspace(min(X_test(:, 2)) - 1, max(X_test(:, 2)) + 1, 200);
[x1Grid, x2Grid] = meshgrid(x1Range, x2Range);
gridData = [x1Grid(:), x2Grid(:)];

% Predict using MLP for the grid points
gridPred = net(gridData'); % Predict probabilities
gridPredClass = reshape(sign(gridPred), size(x1Grid)); % Reshape to match grid

% Scatter plot of test data
figure;
scatter(x0_test(:,1),x0_test(:,2));
hold on 
scatter(x1_test(:,1),x1_test(:,2));

% Overlay decision boundary
contour(x1Grid, x2Grid, gridPredClass, [0 0], 'k', 'LineWidth', 2, 'DisplayName', 'Decision Boundary');

% Add minimal labels and legend
title('Test Data with MLP Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class -1', 'Class 1', 'Decision Boundary', 'Location', 'Best');
hold off;

%% DATA GENERATE FUNCTION
function x = generate_class_samples(n_samples, r)
    % Generate angles uniformly between -pi and pi
    theta = unifrnd(-pi,pi,[n_samples,1]);
    
    % Generate Gaussian noise
    noise = normrnd(0,1,[n_samples, 2]);
    
    % Compute 2D points with noise
    x = zeros(n_samples,2);
    x(:,1) = r.*cos(theta) + noise(:,1);
    x(:,2) = r.*sin(theta) + noise(:,2);
end