function W = LDA_multiclass(X, labels)

%% Fisher Linear Discriminant Analysis:

    % Input:
    %   X:      d x n matrix of original samples
    %           d --- dimensionality of original samples
    %           n --- [# samples] 
    %   labels: n x 1
    %           n --- dimensional vector of class labels
    
    % Output:
    %   W:      d x r transformation matrix 
    %           r --- [#classes - 1]
        
    % Initialization:
    Classes = unique(labels);
    C = numel(Classes);
    n = zeros(C, 1);

    %Initialized as cell due to dimensionality modifications:
    mu = cell(C, 1); 
    Sw = zeros(size(X, 2));
    Sb = zeros(size(X, 2), size(X, 2));

    for i = 1 : C
        % Single class indexes:
        idx = find(labels == Classes(i));
        n(i) = numel(idx);

        % Class mean vector:
        mu{i} = mean(X(idx, :));

        % Within-Class Scatter Matrix:
        Sw = Sw + (n(i) - 1) * cov(X(idx, :));
    end
    
    % Concatenate all mu vectors to compute the overall mean:
    mu_total = mean(cell2mat(mu));

    for i = 1 : C
        % Between-Class Scatter Matrix:
        Sb = Sb + n(i) * (mu{i} - mu_total)' * (mu{i} - mu_total);
    end

    %% Optimization

    % To reduce dimentionality, we need to find W that maximize the ratio
    % of Sb/Sw.
    % W has to satisfy:
    % - (1) Distance between the class means: the larger the better: Sb
    % - (2) The variance of each class: the smaller the better: Sw
    % Thus J(W), the scalar objective function is proportional to Sb and
    % inversely proportional to Sw.
    % This problem is converted to an Eigenvector problem
    % where W = [W1|W2|...|W_c-1] = argmax|W' Sb W|/|W' Sw W|

    [V, D] = eig(Sw \ Sb);
    
    % Sort D (which is a matrix of eigenvectors) and then select 
    % the top vectors assocaited with the top eigenvalues:
    [~, idx] = sort(diag(D), 'descend');
    V_sorted = V(:, idx);
    
    % Select the projection matrix W
    W = V_sorted(:, 1 : C-1);
    
    % Normalize columns of W
    W = W ./ sqrt(sum(W.^2, 1));

end

