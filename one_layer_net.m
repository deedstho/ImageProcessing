function res = big_net(x, w, b, p, s, d, dzdy)

%Forward Pass
res.x1 = x;
res.x2 = vl_nnconv(res.x1, w, b,'pad', p, 'stride', s, ...
      'dilate', d);
  
%Backward Pass
% Backward pass (only if passed output derivative)
if nargin > 6
  res.dzdx2 = dzdy;
  [res.dzdx1, res.dzdw, res.dzdb] = ...
    vl_nnconv(res.x1, w, b, res.dzdx2, 'pad', p, 'stride', s, 'dilate', d) ;
end
  