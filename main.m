clc;
clear;
close all;

addpath('data');

filePath = 'data/column_3c.dat';

feature_names = {
    'pelvic incidence',
    'pelvic tilt',
    'lumbar lordosis angle',
    'sacral slope',
    'pelvic radius',
    'degree spondylolisthesis'
};

class_names = {'DH', 'SL', 'NO'};

[X, y, labels_text] = load_data(filePath, class_names, ' ');


y = y(:)';

if size(X, 1) ~= length(feature_names)
    X = X';
end

[A, B] = size(X);

numClasses = 3;
M = 34;          
sigma = 1;       
acertos = 0;

Ybin = zeros(numClasses, B);
for j = 1:B
    Ybin(y(j), j) = 1;
end
y_pred = zeros(1, B);


for i = 1:B


    x_test = X(:, i);
    y_test = Ybin(:, i);
    x_train = X;
    x_train(:, i) = [];
    y_train = Ybin;
    y_train(:, i) = [];
    numTrain = size(x_train, 2);



    [x_train_norm, media_treino, desvio_treino] = zscore(x_train, 0, 2);
    desvio_treino(desvio_treino == 0) = 1;
    x_test_norm = (x_test - media_treino) ./ desvio_treino;
    M_atual = min(M, numTrain);
    indicesCentros = randperm(numTrain, M_atual);
    C = x_train_norm(:, indicesCentros);
    F = zeros(M_atual, numTrain);

    for j = 1:numTrain
        for k = 1:M_atual
            v = norm(x_train_norm(:, j) - C(:, k));
            F(k, j) = exp(-(v^2) / (2 * sigma^2));
        end
    end

    Fb = [-ones(1, numTrain); F];
    N = y_train * pinv(Fb);
    f_test = zeros(M_atual, 1);

    for j = 1:M_atual

        v = norm(x_test_norm - C(:, j));
        f_test(j) = exp(-(v^2) / (2 * sigma^2));

    end

    f_test = [-1; f_test];

    y_chapeu = N * f_test;
    [~, classe_predita] = max(y_chapeu);
    [~, classe_real] = max(y_test);
    y_pred(i) = classe_predita;
    if classe_predita == classe_real
        acertos = acertos + 1;
    end

end

acc = acertos / B;

fprintf('\nAcuracia final: %.4f\n', acc);
fprintf('Acuracia em porcentagem: %.2f%%\n', acc * 100);

fprintf('\nAmostra\tClasse Real\tClasse Predita\n');

for i = 1:B
    fprintf('%d\t%d\t\t%d\n', i, y(i), y_pred(i));
end
