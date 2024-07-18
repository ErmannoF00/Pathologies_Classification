function [w] = ldaclass_one(X, W, classlabels)
%LDACLASS_ONE design a one-dimensional classifier

is1 = find(W == classlabels(1));
is2 = find(W == classlabels(2));

mu{1} = mean(X(is1, :));
mu{2} = mean(X(is2, :));

N1 = length(is1);
N2 = length(is2);
Sw1 = (N1-1) * cov(X(is1, :)); 
Sw2 = (N2-1) * cov(X(is2, :));
Sw = Sw1 + Sw2; % d x d

% LDA vector
w = inv(Sw) * (mu{1} - mu{2})';  % dx1
w = w ./sqrt(sum(w.^2));
end