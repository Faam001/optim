function [X, FxVal, Iters] = powell_opt(N, X, Eps_Fx, Eps_Step, MaxIter, Fx)

Iters = 0;
f1 = feval(Fx, X, N);
X1 = X;
S = eye(N+1,N);

bGoOn = true;

while bGoOn
  S(N+1,:) = 0; % reset row N+1

  for i= 1:N
    alpha = 0.1;
    alpha = linsearch(X, N, alpha, S, i, Fx);
    X = X + alpha * S(i,:);
    S(N+1,:) = S(N+1,:) + alpha * S(i,:);
  end

  alpha = 0.1;
  alpha = linsearch(X, N, alpha, S, N+1, Fx);
  X = X + alpha * S(N+1,:);
  X2 = X;

  f2 = feval(Fx, X2, N);

  if abs(f2 - f1) < Eps_Fx
    break;
  end

  if norm(X2 - X1) < Eps_Step
    break
  end

  Iters = Iters + 1;

  if Iters >= MaxIter
    break
  end

  X1 = X2;
  for k=1:N
    for m=1:N
      S(k, m) = S(k+1,m);
    end
  end

end

FxVal = feval(Fx, X, N);

function y = FxEx(N, X, S, ii, alpha, Fx)

  X = X + alpha * S(ii,:);
  y = feval(Fx, X, N);

% end

function alpha = linsearch(X, N, alpha, S, ii, Fx)

  MaxIt = 100;
  Toler = 0.0001;

  iter = 0;
  bGoOn = true;
  while bGoOn
    iter = iter + 1;
    if iter > MaxIt
      alpha = 0;
      break
    end

    h = 0.01 * (1 + abs(alpha));
    f0 = FxEx(N, X, S, ii, alpha, Fx);
    fp = FxEx(N, X, S, ii, alpha+h, Fx);
    fm = FxEx(N, X, S, ii, alpha-h, Fx);
    deriv1 = (fp - fm) / 2 / h;
    deriv2 = (fp - 2 * f0 + fm) / h ^ 2;
    if deriv2 == 0
        break
    end
    diff = deriv1 / deriv2;
    alpha = alpha - diff;
    if abs(diff) < Toler
      bGoOn = false;
    end
  end

% end